- Non usare mai pip direttamente — tutto passa da uv add / uv remove, così il pyproject.toml rimane sempre sincronizzato con quello che hai installato.
- uv run invece di attivare il venv — invece di source .venv/bin/activate (che tra l'altro su Windows è diverso), usi direttamente uv run python script.py o uv run jupyter lab. Più pulito e meno error-prone.
- uv sync — se cloni un progetto esistente che ha già un pyproject.toml, basta uv sync per ricreare l'ambiente identico. Utile quando cambi macchina o condividi codice.
- uv lock — UV genera automaticamente un uv.lock che fissa le versioni esatte di ogni dipendenza. Non toccarlo a mano, ma committalo su git se lavori in team.


## Setup progetto DS

``` powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

uv py list
uv python install 3.13
uv init studio-paper
uv add pandas numpy matplotlib

mkdir notebooks 
mkdir src
uv add --group dev ruff jupyterlab pytest

```

## Setup pyspark
``` powershell
winget install Microsoft.OpenJDK.17

# di solito le variabili sono impostate automaticamente
Get-ChildItem "C:\Program Files\Microsoft" -Filter "*jdk*" -Directory
[System.Environment]::SetEnvironmentVariable("JAVA_HOME", "C:\Program Files\Microsoft\jdk-17.0.19.10-hotspot", "Machine")
$currentPath = [System.Environment]::GetEnvironmentVariable("PATH", "Machine")
[System.Environment]::SetEnvironmentVariable("PATH", "$currentPath;C:\Program Files\Microsoft\jdk-17.0.19.10-hotspot\bin", "Machine")

uv add pyspark

```

Limita le risorse con

``` python
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder \
    .master("local[2]") \  # max 2 core, non tutti
    .config("spark.driver.memory", "1g") \
    .config("spark.executor.memory", "1g") \
    .appName("poc") \
    .getOrCreate()
```


## Setup claude

``` powershell
irm https://claude.ai/install.ps1 | iex
# Spesso le variabili sono già settate
[Environment]::SetEnvironmentVariable("PATH", "$env:PATH;$env:USERPROFILE\.local\bin", [EnvironmentVariableTarget]::User)
```