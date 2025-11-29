"""Performance timer with section tracking."""

import time
from contextlib import contextmanager
from typing import Dict


class Timer:
    """Performance timer with hierarchical section tracking."""
    
    def __init__(self):
        self.times = {}
        self.start_times = {}
    
    def start(self, name: str):
        """
        Start timing a section.
        
        Args:
            name: Section name
        """
        self.start_times[name] = time.time()
    
    def stop(self, name: str) -> float:
        """
        Stop timing a section.
        
        Args:
            name: Section name
        
        Returns:
            Elapsed time in seconds
        """
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.times[name] = elapsed
            del self.start_times[name]
            return elapsed
        return 0.0
    
    @contextmanager
    def time_section(self, name: str):
        """
        Context manager for timing a code block.
        
        Args:
            name: Section name
        """
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)
    
    def get_times(self) -> Dict[str, float]:
        """Get all recorded times."""
        return self.times.copy()
    
    def reset(self):
        """Reset all timers."""
        self.times.clear()
        self.start_times.clear()
    
    def summary(self) -> str:
        """Generate timing summary string."""
        lines = ["Timing Summary:"]
        lines.append("-" * 50)
        
        total = self.times.get('total', sum(self.times.values()))
        
        for name, elapsed in sorted(self.times.items()):
            if name != 'total':
                pct = (elapsed / total * 100) if total > 0 else 0
                lines.append(f"  {name:25s}: {elapsed:8.3f} s ({pct:5.1f}%)")
        
        lines.append("-" * 50)
        lines.append(f"  {'TOTAL':25s}: {total:8.3f} s")
        
        return "\n".join(lines)
