import os
from abc import abstractmethod


class ClusterResolver:

    def __init__(self):
        pass

    @property
    @abstractmethod
    def rank(self) -> int:
        """
        Returns the rank of the given process within the communicator
        """
        pass

    @property
    @abstractmethod
    def world_size(self) -> int:
        """
        Returns the number of processes in the communicator
        """
        pass

    @property
    @abstractmethod
    def local_rank(self) -> int:
        """
        Returns the rank of the given process within the node
        """
        pass

    @property
    @abstractmethod
    def master(self) -> bool:
        """
        Returns whether or not the given process is considered the master
        """
        pass

    @property
    @abstractmethod
    def master_addr(self) -> str:
        """
        Returns the address of the master for Pytorch to setup distribution
        """
        pass

    @property
    @abstractmethod
    def master_port(self) -> int:
        """
        Returns the port on the master node (should avoid port conflict)
        """
        pass


class SlurmClusterResolver(ClusterResolver):

    def __init__(self):
        super().__init__()
        self.reference_port = 12345
        self.rank = int(os.environ["SLURM_PROCID"])
        self.world_size = int(os.environ["SLURM_NTASKS"])
        self.local_rank = int(os.environ["SLURM_LOCALID"])
        self.master = self.rank == 0
        self.master_addr = self.get_first_host(os.environ["SLURM_JOB_NODELIST"])

        # to avoid port conflict on the same node
        self.master_port = self.reference_port + int(min(os.environ['SLURM_STEP_GPUS'].split(",")))

    @staticmethod
    def get_first_host(hostlist: str) -> str:
        from re import findall, sub, split
        regex = "\[([^[\]]*)\]"
        all_replacement: list[str] = findall(regex, hostlist)
        new_values = [split("-|,", element)[0] for element in all_replacement]
        for i in range(len(new_values)):
            hostlist = sub(regex, new_values[i], hostlist, count=1)
        return hostlist.split(",")[0]
