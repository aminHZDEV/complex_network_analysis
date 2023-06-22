from Utils.convert_xlsx_to_csv import Converter

converter = Converter(from_file_path="Datasets", to_file_path="Datasets")

converter.convertXlsxToCsv()
