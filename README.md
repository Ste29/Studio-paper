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