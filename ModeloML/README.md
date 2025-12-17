# LSChML

Scripts para capturar secuencias de manos, entrenar un modelo LSTM y ejecutar inferencia en tiempo real (LSCh).

## Instalación (Windows)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

Uso

Capturar datos:
python [CapturarSecuencias.py](http://_vscodecontentref_/5)

Entrenar:
python [EntrenarModeloSeñas.py](http://_vscodecontentref_/6)

Inferencia:
python [InferenciaSeñas.py](http://_vscodecontentref_/7)

Nota: el dataset se guarda en SecuenciasSeñas/ (ignorado por git).