# Survey Anomaly Detection

Я реализовал поиск аномалий в исследовательских данных с помощью робастных статистик, обработки Parquet и объяснения причин обнаруженных отклонений.

## Запуск

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python solution_NikitinAntonPavlovich_SHCT-111.py --help
```

## Устройство проекта

CLI-параметры доступны через `--help`; подробности алгоритма — в `algorithm_description.md`. Исследовательские данные с демографическими сведениями исключены. Для работы нужен разрешённый набор Parquet.

Я публикую исходный код без локальных паролей, окружений, баз и журналов. Для воспроизведения анализа я указываю необходимые данные и зависимости.

## Автор

Антон Никитин — [anton-nikitin21](https://github.com/anton-nikitin21).
