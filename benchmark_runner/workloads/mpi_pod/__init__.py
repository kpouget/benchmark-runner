import os, sys
import urllib3

from pathlib import Path
import yaml
import time
import json

import kubernetes.client
import kubernetes.config
import kubernetes.utils

from kubernetes.client import V1ConfigMap, V1ObjectMeta

from benchmark_runner.common.elasticsearch.elasticsearch_exceptions import ElasticSearchDataNotUploaded
from benchmark_runner.workloads.workloads_operations import WorkloadsOperations
from benchmark_runner.common.logger.logger_time_stamp import logger_time_stamp, logger


kubernetes.config.load_kube_config()

v1 = kubernetes.client.CoreV1Api()
appsv1 = kubernetes.client.AppsV1Api()
batchv1 = kubernetes.client.BatchV1Api()
customv1 = kubernetes.client.CustomObjectsApi()

k8s_client = kubernetes.client.ApiClient()

def initialize_workload(self):
    self._oc.delete_all_resources(["MPIJobs", "Pods"])
    self.odf_pvc_verification()
    self._template.generate_yamls()
    self.start_prometheus()

def finalize_workload(self):
    self.end_prometheus()
    self.upload_run_artifacts_to_s3()

class MpiPod(WorkloadsOperations):
    """
    This class run vdbench pod
    """
    def __init__(self):
        super().__init__()
        self.name = ''
        self.workload_name = ''
        self.es_index = ''
        self.status = ''
        self.pod_name = None
        self.workload = "mpi_pod"
        self.kind = "pod"
        self.resource_path = Path(self._run_artifacts_path) / f"{self.workload}.yaml"
        self.mpijob_name = None

    def run(self):
        self.pod_name = f'{self.workload}-{self._trunc_uuid}'

        self.es_index = 'mpi-results'
        self._environment_variables_dict['kind'] = self.kind

        try:
            version_dict = customv1.get_cluster_custom_object("config.openshift.io", "v1",
                                                              "clusterversions", "version")
            print("Connected to the cluster.")
        except Exception as e:
            print(f"ERROR: {e}")
            print("WARNING: Is the Kubernetes cluster reachable? Aborting.")

            return False

        self._do_cleanup()

        try:
            if self._do_launch() == False:
                print("Launch failure, aborting")
                return False
        except Exception as e:
            print("Caught an exception during launch time, aborting.")
            raise e

        try:
            wait_success = self._do_wait()
        except KeyboardInterrupt as err:
            print("\nKeyboard Interrupted :/")
            sys.exit(1)

        except Exception as err:
            raise err

        success = (
            wait_success,
            self._do_save_mpi_artifacts(),
            self._do_save_system_artifacts(),
            self._publish(),
        )

        if False not in success:
            return True

        print("Failure detected ...", success)
        return False

    def _do_cleanup(self):
        namespace = self._environment_variables_dict["namespace"]
        first = True
        while True:
            pods = v1.list_namespaced_pod(namespace=namespace)
            if len(pods.items) == 0:
                if not first:
                    print("Done.")
                break

            if first:
                print("Found pods still alive ...")
                first = False

            deleting_pods = []
            for pod in pods.items:
                try:
                    v1.delete_namespaced_pod(namespace=namespace, name=pod.metadata.name)
                    deleting_pods.append(pod.metadata.name)
                except kubernetes.client.exceptions.ApiException as e:
                    if e.reason != "Not Found": raise e

            print(f"Deleting {len(deleting_pods)} Pods:", " ".join(deleting_pods))
            time.sleep(5)

        pass

    def _do_launch(self):
        self._oc._create_async(str(self.resource_path))

        with open(self.resource_path) as f:
            self.docs = list(yaml.safe_load_all(f))
        self.mpijob_name = self.docs[0]["metadata"]["name"]

        return True

    def _do_wait(self):
        namespace = self._environment_variables_dict["namespace"]

        print(f"Waiting for {self.mpijob_name} to complete its execution ...")
        failed = False
        namespace = self._environment_variables_dict["namespace"]
        while not failed:
            print(".", end="", flush=True)
            time.sleep(5)
            mpijob = customv1.get_namespaced_custom_object("kubeflow.org", "v1",
                                                           namespace, "mpijobs", self.mpijob_name)
            status = mpijob.get("status")
            if status and status.get("completionTime"):
                break

            pods = v1.list_namespaced_pod(namespace=namespace,
                                      label_selector=f"training.kubeflow.org/job-name={self.mpijob_name}")
            for pod in pods.items:
                try:
                    if pod.status.container_statuses[0].state.waiting.reason == "ImagePullBackOff":
                        print(f"\nERROR: Pod {pod.metadata.name} is in state ImagePullBackOff. Aborting")
                        failed = True
                except Exception: pass

                try:
                    if pod.status.container_statuses[0].state.terminated.reason == "Error":
                        print(f"\nERROR: Pod {pod.metadata.name} is in state Error. Aborting")
                        failed = True
                except Exception: pass

        if failed:
            print("Failed")
            return False

        print("\nDone")

    def _do_save_mpi_artifacts(self):
        namespace = self._environment_variables_dict["namespace"]
        failed = False

        def save_yaml(obj, filename):
            obj_dict = obj if isinstance(obj, dict) else obj.to_dict()

            with open(Path(self._run_artifacts_path) / filename, "w") as out_f:
                yaml.dump(obj_dict, out_f)

        mpijob = customv1.get_namespaced_custom_object("kubeflow.org", "v1",
                                                       namespace, "mpijobs", self.mpijob_name)
        save_yaml(mpijob, "mpijob.status.yaml")

        pods = v1.list_namespaced_pod(namespace=namespace,
                                      label_selector=f"training.kubeflow.org/job-name={self.mpijob_name}")
        save_yaml(pods, "pods.status.yaml")
        for pod in pods.items:
            phase = pod.status.phase

            print(f"{pod.metadata.name} --> {phase}")
            try:
                logs = v1.read_namespaced_pod_log(namespace=namespace, name=pod.metadata.name)
            except kubernetes.client.exceptions.ApiException as e:
                print(f"ERROR: could not get Pod {pod.metadata.name} logs:", json.loads(e.body))
                logs = str(e)

            with open(Path(self._run_artifacts_path) / f"pod.{pod.metadata.name}.log", "w") as out_f:
                print(logs, end="", file=out_f)

            if pod.metadata.labels["training.kubeflow.org/job-role"] == "launcher":
                if phase != "Succeeded":
                    failed = True
            elif pod.metadata.labels["training.kubeflow.org/job-role"] == "worker":
                if phase == "Error":
                    failed = True

        return not failed

    def _do_save_system_artifacts(self):
        print("-----")
        print("Collecting artifacts ...")

        def save_nodes():
            nodes = v1.list_node()
            nodes_dict = nodes.to_dict()

            try: del nodes_dict["metadata"]["managed_fields"]
            except KeyError: pass # ignore

            try: del nodes_dict["status"]["images"]
            except KeyError: pass # ignore

            with open(Path(self._run_artifacts_path) / "nodes.yaml", "w") as out_f:
                yaml.dump(nodes_dict, out_f)

        def save_cluster_version():
            print("Saving OpenShift version ...")

            version_dict = customv1.get_cluster_custom_object("config.openshift.io", "v1",
                                                              "clusterversions", "version")
            try: del version_dict["metadata"]["managedFields"]
            except KeyError: pass # ignore

            with open(Path(self._run_artifacts_path) / f"nodes.yaml", "w") as out_f:
                yaml.dump(version_dict, out_f)

        save_nodes()
        save_cluster_version()
        print("Artifacts saved in", self._run_artifacts_path)

    def _publish(self):
        print("Nothing to publish yet :/")

        return

        if not self._es_host: return

        # to be coded

        self._upload_to_elasticsearch(index=self.es_index,
                                      kind=self.kind,
                                      status=self.status,
                                      result=result)

        # verify that data upload to elastic search according to unique uuid
        self._verify_elasticsearch_data_uploaded(index=self.__es_index,
                                                 uuid=self._uuid)
