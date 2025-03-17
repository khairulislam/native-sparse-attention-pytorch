import os, torch
import psutil, os
from timeit import default_timer as timer
import datetime

def running_on_windows():
    return os.name == 'nt'

class Benchmark:
    def __init__(self, device="cuda", log_file="benchmark_results.log"):
        self.device = device
        self.process = psutil.Process()  # Get current process for RAM tracking
        self.log_file = log_file

    def start(self):
        """Start tracking time, GPU memory, and RAM usage."""
        torch.cuda.synchronize() if self.device.startswith("cuda") else None
        torch.cuda.reset_peak_memory_stats() if self.device.startswith("cuda") else None  # Reset peak memory tracking
        self.start_time = timer() # time.time()
        self.start_ram = self.process.memory_info().rss / 1e6  # RAM in MB
        self.start_gpu_mem = self._get_gpu_memory() if self.device.startswith("cuda") else 0

    def stop(self):
        """Stop tracking and return elapsed time, GPU memory, and RAM usage."""
        torch.cuda.synchronize() if self.device.startswith("cuda") else None
        elapsed_time = timer() - self.start_time
        ram_used = (self.process.memory_info().rss / 1e6) - self.start_ram  # RAM in MB
        gpu_used = self._get_gpu_memory() - self.start_gpu_mem if self.device.startswith("cuda") else 0
        peak_gpu_mem = self._get_peak_gpu_memory() if self.device.startswith("cuda") else 0  # Get peak memory

        results = (
            f"\n[Benchmark Results - {datetime.datetime.now()}]\n"
            f"⏳ Time Elapsed: {elapsed_time:.4f} sec\n"
            f"🖥️  RAM Used: {ram_used:.2f} MB\n"
        )
        if self.device.startswith("cuda"):
            results += (
                f"🖥️  GPU Memory Used: {gpu_used:.2f} MB\n"
                f"🚀 Peak GPU Memory: {peak_gpu_mem:.2f} MB\n"
            )

        print(results)
        self._log_results(results)

        return elapsed_time, ram_used, gpu_used, peak_gpu_mem

    def _get_gpu_memory(self):
        """Returns current GPU memory usage in MB using PyTorch."""
        return torch.cuda.memory_allocated() / 1e6  # Convert bytes to MB

    def _get_peak_gpu_memory(self):
        """Returns peak GPU memory usage in MB using PyTorch."""
        return torch.cuda.max_memory_allocated() / 1e6  # Convert bytes to MB

    def _log_results(self, results):
        """Logs the benchmark results to a file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(results + "\n" + "-" * 50 + "\n")
