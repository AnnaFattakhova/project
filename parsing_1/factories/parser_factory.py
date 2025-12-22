# Шаблон проектирования: FACTORY METHOD
# Фабрика, создающая подходящий парсер по имени источника

from ..parser import OtzyvruParser


class ParserFactory:
    """
    Фабрика для создания парсеров разных источников (otzyvru, flamp, irecommend...)
    Это реализация шаблона Factory Method.
    """

    @staticmethod
    def create_parser(source: str, url: str, save_path: str, log_path: str):
        """
        Возвращает объект парсера в зависимости от типа источника.
        """
        source = source.lower().strip()

        if source == "otzyvru":
            return OtzyvruParser(url, save_path, log_path)

        # здесь можно расширять (для других парсеров)

        raise ValueError(f"Неизвестный тип источника: {source}")
