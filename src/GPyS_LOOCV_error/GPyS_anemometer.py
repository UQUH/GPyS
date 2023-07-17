from click import progressbar
from os import walk, path, makedirs
from re import search
from datetime import datetime
from json import dumps
import csv

"""Regex (regular expressioin) meaning
(some_regex) : group the regex match
		  \d : digit
		   ? : zero or one (referring to the previous)
		   ^ : start with
		   . : any character (a to z and A to Z)
		   + : one or more (referring to the previous)
		   - : literal hyphen
		  \. : escape . with backslash to imply literal dot character
		   $ : end with

	string = "^.+-train-.+\.csv$" means :
		if string starts with one or more character
		followed by an hyphen character
		followed by the word "train"
		followed by an hyphen character
		followed by one or more character 
		and ends with ".csv"
		then we have a match
"""


class Anemometer:

    @staticmethod
    def get_X_and_theta(
            path_to_files,
            file_filter,
            no_of_files_to_work_with=-1,  # -1 => use of files
            filter_file_from_position_start=None,  # None => start from file 0
            filter_file_from_position_end=None,  # None => goes to the end of file
            no_of_rows_in_files_to_work_with=-1,  # -1 => use of rows in files
            merge_all_x=False  # Should we convert all files content to a 2-D or 3-D
    ):
        if no_of_files_to_work_with == 0 or no_of_rows_in_files_to_work_with == 0:
            return [], []

        sheet_names_and_parameters = \
            Anemometer.get_sheet_names_and_parameters(
                path_to_files,
                file_filter,
                filter_file_from_position_start,
                filter_file_from_position_end
            )

        all_Xs = []
        parameters = []

        for sheet_name, parameter in sheet_names_and_parameters.items():
            rows = []
            full_path = path.join(path_to_files, sheet_name)

            no_of_rows_in_files_to_work_with_help = no_of_rows_in_files_to_work_with

            with open(full_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)

                print(f"\nReading '{sheet_name}' file...")
                with progressbar(reader, fill_char="\u2588") as bar:
                    for row_data in bar:
                        row = list(map(float, list(row_data.values())[1:]))
                        rows.append(row)

                        no_of_rows_in_files_to_work_with_help -= 1
                        if no_of_rows_in_files_to_work_with_help == 0:
                            break

                    all_Xs.append(rows)
                    parameters.append([parameter])

            no_of_files_to_work_with -= 1
            if no_of_files_to_work_with == 0:
                break

        merged_Xs = []
        if merge_all_x:
            print(f"\n\nMerging all '{file_filter}' csv files...")
            with progressbar(zip(*all_Xs), fill_char="\u2588") as bar:
                for Xs in bar:
                    merged = sum(Xs, [])
                    merged_Xs.append(merged)

        return (merged_Xs if merge_all_x else all_Xs), parameters

    @staticmethod
    def get_sheet_names_and_parameters(
            path_to_files,
            file_filter,
            filter_file_from_position_start=None,
            filter_file_from_position_end=None
    ):
        excel_sheet_names = Anemometer.__get_X_files(path_to_files, file_filter)

        if filter_file_from_position_start == None:
            filter_file_from_position_start = 0

        if filter_file_from_position_end == None:
            filter_file_from_position_end = len(excel_sheet_names) - 1

        print(f"\nDetecting '{file_filter}' parameters from csv files...")
        with progressbar(excel_sheet_names, fill_char="\u2588") as bar:
            sheet_names_and_parameters_unsorted = [
                (sheet_name, float(match.group(1) if match.group(1) else 0))
                for sheet_name in bar
                if (match := search("-((?:-|)\d+\.?\d+)\.csv$", sheet_name))
            ]

            print(f"\n\nSorting '{file_filter}' csv files by parameters...")
            sheet_names_and_parameters_unsorted.sort(key=lambda x: x[1])
            with progressbar(
                    sheet_names_and_parameters_unsorted[
                    filter_file_from_position_start:filter_file_from_position_end + 1
                    ],
                    fill_char="\u2588"
            ) as bar:
                sheet_names_and_parameters_sorted = {
                    sheet_name: parameter
                    for sheet_name, parameter in bar
                }

                return sheet_names_and_parameters_sorted

    @staticmethod
    def __get_X_files(path_to_files, file_filter):
        _, _, all_file_names = next(walk(path_to_files))

        print(f"\nDetecting '{file_filter}' csv files from {path_to_files}...")
        with progressbar(all_file_names, fill_char="\u2588") as bar:
            excel_sheet_names = [
                file_name
                for file_name in bar
                if search(f"^.+-{file_filter}-.+\.csv$", file_name)
            ]
            return excel_sheet_names

    @staticmethod
    def write_to_csv(
            rows,
            path_to_files="./GPyS_saved/",
            file_name="",
            cols_notation="V",
            should_include_serial_no=True,
            serial_no_title=""
    ):
        makedirs(path.dirname(path_to_files), exist_ok=True)

        today = datetime.utcnow().strftime("%Y%m%d_%H%M")
        file_name = file_name if file_name else f"saved-{today}"
        full_path = path.join(path_to_files, f"{file_name}.csv")

        print(f'\n\nWriting to "{full_path}" started...')

        with open(full_path, 'w', newline='') as csvfile:
            no_of_col = len(rows[0])
            field_names = [serial_no_title] if should_include_serial_no else [""]
            field_names += [f"{cols_notation}{c}" for c in range(1, no_of_col + 1)]
            writer = csv.DictWriter(csvfile, fieldnames=field_names)

            writer.writeheader()

            # No print statement should occur within this progressbar block of code.
            with progressbar(
                    rows,
                    fill_char="\u2588",
                    empty_char="-",
                    show_eta=False,  # Don't show time in %(info)s
                    # show_percent=True,
                    # show_pos=True, # Show row counting in %(info)s
                    # bar_template='[%(bar)s]%(info)s',
            ) as bar:
                for r, row in enumerate(bar, 1):
                    record = {serial_no_title: r} if should_include_serial_no else {"": ""}
                    record.update({
                        f"{cols_notation}{c}": cell_value
                        for c, cell_value in enumerate(row, 1)
                    })

                    writer.writerow(record)
        print("Writing completed!!!")


if __name__ == "__main__":
    path_to_files = "./GPyS_Anemometer_Predict/"

    X, theta = Anemometer.get_X_content(
        path_to_files=path_to_files,
        file_filter="train",
        no_of_files_to_work_with=1,
        no_of_rows_in_files_to_work_with=1,
    )
# print(X)
# print(dumps(X, indent=4, ensure_ascii=False, default=str))