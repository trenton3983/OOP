import json
from glob import glob
from pathlib import Path, WindowsPath
from natsort import natsorted


class JSONProcessing:
    def __init__(self, glob_pattern, save_name):
        """

        :param glob_pattern:
        :type glob_pattern: str
        :param save_name: name for the final csv file
        :type save_name: str
        """
        self.glob_pattern = Path(glob_pattern)
        self.save_name = save_name

    def process_path(self):
        """

        :return:
        """
        files_testrun = natsorted(glob(str(self.glob_pattern)))
        files_dict = {}
        for files in files_testrun:
            dir_capture = Path(files).parents[0].joinpath(r'waveforms\*\capture.json')
            files_capture = natsorted(glob(str(dir_capture)))
            files_dict[files] = files_capture
        return files_dict

    def read_json(self, filename):
        """
        reads json files
        :param filename:
        :return:
        """
        with open(filename, 'rb') as json_data:
            r = json.loads(json_data.read())
        return r

    def glob_data_files(self, file_dir) -> list:
        """
        Given a directory to the capture data, finds, sorts and returns all the *.bin file paths
        :param file_dir:
        :type file_dir: str
        :return files_capture: list of str
        """
        dir_capture = Path(file_dir).parents[0].joinpath(r'*.bin')
        files_capture = natsorted(glob(str(dir_capture)))
        return files_capture

    def status_contents(self, file_dir):
        """
        Parses the status.json files into headers and values
        :param file_dir:
        :return header, values: list
        """
        r = self.read_json(file_dir)
        header = ['testrun_status']
        values = [r['Status']]
        return header, values

    def testrun_contents(self, file_dir):
        """
        Parses the testrun.json files into headers, values and test_points
        :param file_dir: location of the testrun.json file
        :type file_dir: str
        :return headers, values and test_points: lists
        """
        r = self.read_json(file_dir)
        header = list(r.keys())
        values = list(r.values())
        values.pop(-1)
        test_points = r['Test Points']

        fixed_tp = []
        for tp in test_points:
            if len(tp) != 0:
                if '\n' in tp:
                    tp = tp.split('\n')[0]
                fixed_tp.append(tp)
        test_points = natsorted(fixed_tp)
        return header, values, test_points

    def meta_extract(self, meta):
        """
        meta is a section of the capture.json file and it contains the power supply information.
        this method gives each of the four power supplies a unique name.  the names are added to
        the header and the values for each of the power supply attributes is added to the values
        :param meta: json formatted data for powersupplies
        :type meta: json
        :return header: list
        :return values: list
        """
        meta_data = json.loads(meta)
        meta_keys = meta_data.keys()
        header = list(meta_keys)[:-1]
        values = list(meta_data.values())[:-1]
        power_supply = meta_data['Power Supply']
        for i, supply in enumerate(power_supply):
            header_ = list(supply.keys())
            values_ = list(supply.values())

            header_ = [(x + f'_{i+1}') for x in header_]

            header.extend(header_)
            values.extend(values_)
        return header, values

    def capture_contents(self, file_dir):
        """
        Receives the path to a capture.json file and extracts the headers and values, which are returned as lists.
        :param file_dir:
        :return header:
        :return values:
        """
        r = self.read_json(file_dir)
        r_keys = list(r.keys())
        header = r_keys[0:2]  # Remove meta, names & compress
        header.append('compress')
        values = list(r.values())[:2]  # Remove meta, names & compress
        values.append(r['compress'])
        for k in r_keys:
            if k != 'meta':
                pass
            elif k == 'meta':
                meta_header, meta_values = self.meta_extract(r[k])
        header.extend(meta_header)
        values.extend(meta_values)
        return header, values

    def main(self):
        """
        Creates a csv of all the processed testrun, status and capture json files.
        Each row of the final file represents one measured testpoint.
        """
        files_dict = self.process_path()
        with open(self.save_name, 'w') as csv_file:
            header_count = 0
            for k, v in files_dict.items():

                status_file_path = Path(k).parents[0].joinpath('status.json')

                tr_header, tr_values, test_points = self.testrun_contents(k)
                st_header, st_values = self.status_contents(status_file_path)

                if bool(v):  # does the list v have contents

                    for i in v:
                        # i is the address to each capture.json file under k
                        data_files = self.glob_data_files(i)

                        cap_header, cap_values = self.capture_contents(i)
                        header = ['collection_status'] + st_header + tr_header + cap_header + ['data_address']
                        if header_count == 0:
                            #                 print(header)
                            header_str = ','.join(str(h) for h in header)
                            csv_file.write(header_str)
                            csv_file.write('\n')
                            print(header_str)
                            #                 print(len(header))
                            print('\n')
                            header_count = 1
                        for i_, point in enumerate(test_points):
                            try:
                                values = ['data_collected'] + st_values + tr_values + [point] + \
                                         cap_values + [WindowsPath(data_files[i_])]
                            except IndexError:
                                print(
                                    f"There are {len(test_points)} test points, but only {len(data_files)} data files.")
                                values = ['no_data_collected'] + st_values + tr_values + [point] + \
                                         cap_values + [WindowsPath(data_files[0][:-8])]

                            values_str = r','.join(str(v) for v in values)
                            csv_file.write(values_str)
                            csv_file.write('\n')
                            print(values_str)
                            #                 print(str(WindowsPath(data_files[i_])))
                            print(test_points)
                            print(point)
                            #                 print(values)
                            #                 print(len(values))
                            print('\n')

                else:
                    print('***********Not Capture Data***********')

                print('\n')


if __name__ == '__main__':
    directory_testrun = r'\\npo\coos\LNO_Validation\Validation_Data\_data\Startup\ingot\J76865-003\03\*\*\testrun.json'
    file_save_name = 'class_ingot_test.csv'

    blah = JSONProcessing(directory_testrun, file_save_name)
    blah.main()
