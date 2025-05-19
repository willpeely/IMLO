import os
print(os.cpu_count())  

import psutil
print(psutil.cpu_count(logical=False))  # Physical cores only