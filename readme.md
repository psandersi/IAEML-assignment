# Робо-такси в JAX

Daria Galushko · AI-01

[Модель автомобиля и среда](reports/assignment1.md) · [Обучение и результаты](reports/assignment2.md)

Из корня репозитория в Git Bash под Windows:

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements-assignment.txt
./.venv/Scripts/python.exe -m pytest tests -q
./.venv/Scripts/python.exe -u scripts/guided_run.py
```

Последняя команда выполняет три обучения и записывает результаты в новый каталог `artifacts/guided`. Для повторения одного запуска в другой папке см. второй отчёт. Первоначальный эксперимент сохранён в [artifacts/RESULTS.md](artifacts/RESULTS.md).
