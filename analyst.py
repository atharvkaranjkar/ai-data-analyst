import io
import tempfile
import subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import sys
import os


# db import
try:
    import duckdb
except Exception:
    duckdb = None


#-------------- Load Data--------------

def _looks_like_csv(raw_bytes:bytes) -> bool:
    try:
        sample = raw_bytes[:1024].decode(error="ignore")
    except Exception:
        return False
    return "," in sample and "\n" in sample