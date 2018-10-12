import pandas as pd
from zipfile import ZipFile
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import gc


class AutoFilter:
    def __init__(self, data_frame, tc_list, plot_dir, plot_multi_params=None):
        """
         Given the two class arguments, data_frame and tc_list (test condition list), where tc_list is the
        important column headers of the DataFrame, iterate through the list and create a list of unique
        properties for each tc.  Each dictionary value is accessible with data attributes.  The data attributes
        return a tuple with the test condition at index 0 and the unique values at index 1
        (i.e. ('Serial Number', ['23AC', '2450'])).

        Each of the graph types is a list of tuples which are used in the corresponding method to create the
        required graphs and return the DataFrame for those waveforms.

        tc_list = ['DUT', 'Serial_Number', 'Test_Station', 'Test_Points', 'Slew_1',
                   'Temperature_Setpoint', 'On_1', 'Voltage_1', 'On_2', 'Voltage_2']

        :param data_frame: DataFrame to be plotted
        :type data_frame: pd.DataFrame
        :param tc_list: headers from the DataFrame used as filters
        :type tc_list: list
        :param plot_dir: directory to save the plots
        :type plot_dir: str
        :param plot_multi_params:
        :type plot_multi_params: dict
        """

        cond_dict = {}
        for i, condition in enumerate(tc_list):
            cond_dict[condition] = list(data_frame[condition].unique())
        self.cond_dict = cond_dict

        # Data Attributes
        self.plot_multi_params = plot_multi_params
        self.dut = ('DUT', cond_dict['DUT'])
        self.sn = ('Serial_Number', cond_dict['Serial_Number'])
        self.tp = ('Test_Points', cond_dict['Test_Points'])
        self.slew = ('Slew_1', cond_dict['Slew_1'])
        self.temp = ('Temperature_Setpoint', cond_dict['Temperature_Setpoint'])
        self.ts = ('Test_Station', cond_dict['Test_Station'])
        try:
            self.v1 = ('Voltage_1', cond_dict['Voltage_1'])
        except KeyError:
            print(f'No Voltage_1')
        try:
            self.v2 = ('Voltage_2', cond_dict['Voltage_2'])
        except KeyError:
            print(f'No Voltage_2')
        try:
            self.v3 = ('Voltage_3', cond_dict['Voltage_3'])
        except KeyError:
            print(f'No Voltage_3')
        try:
            self.v4 = ('Voltage_4', cond_dict['Voltage_4'])
        except KeyError:
            print(f'No Voltage_4')

        # Graph Types - i.e. test point and slew or test point and temp
        self.seq_1ps_list = [self.slew, self.v1]
        self.tp_slew_list = [self.tp, self.v1, self.slew]
        self.tp_temp_list = [self.tp, self.v1, self.temp]
        self.five_list = [self.sn, self.tp, self.v1, self.temp, self.slew]

        try:
            self.seq_2ps_list = [self.slew, self.v1, self.v2]
            self.six_list = [self.sn, self.tp, self.v1, self.v2, self.temp, self.slew]
        except AttributeError:
            print('There is no Voltage_2, so six_test_conditions and seq_2ps are disabled.')
            pass
        try:
            self.seven_list = [self.sn, self.tp, self.v1, self.v2, self.v3, self.temp, self.slew]
        except AttributeError:
            print('There is no Voltage_3, so seven_test_conditions is disabled.')
            pass

        self.plot_dir = plot_dir
        self.df = data_frame

    @staticmethod
    def read_waveform(data_address):
        """
        Uses numpy to create an array for the specified waveform
        :param data_address: string -> path and file name
        :return: np.array of floats
        """
        f = ZipFile(os.path.join(data_address), 'r')
        f_name = f.namelist()
        w = f.read(f_name[0])  # the key is the file name without the 'bin' extension (i.e. CH1)
        f.close()
        dt = np.dtype(float)
        dt = dt.newbyteorder('>')
        wf = np.frombuffer(w, dtype=dt)
        return wf

    @staticmethod
    def calculate_time(ix, xi, wf):
        """
        Given initial x and x increment, all the x (time) values
        are calculated for the waveform
        :param ix: initial x - float
        :param xi: x increment - float
        :param wf: waveform - np.array
        :return x_time: np.array x component for each value of the wf
        """
        wf_len = len(wf)
        x_time = np.arange(ix, (ix + wf_len * xi), xi)
        return x_time

    def create_wf_df(self, df_filter):
        """
        Given a DataFrame (i.e. df_filter) this method:
        1) looks at the data_address column
        2) send the address to read_waveform which retuns a 1 by x array of the waveform data points
        3) the array is added to a DataFrame
        4) the waveform, xi and ix is sent to calculate_time which returns an array of time (x) values
        5) a time column is added to the DataFrame for each waveform column
        6) the DataFrame of waveforms and time is returned
        :param df_filter: Pandas.DataFrame
        :return df_wf: Pandas.DataFrame
        """
        df_wf = pd.DataFrame()
        wf_length_count = {}
        for i, v in enumerate(df_filter['data_address']):
            wf = self.read_waveform(v)
            wf_len = len(wf)
            if wf_len not in wf_length_count:
                wf_length_count[wf_len] = 1
            else:
                wf_length_count[wf_len] += 1
            if i == 0:
                first_wf_len = wf_len  # length of first waveform
            x_time = self.calculate_time(ix=list(df_filter['initial_x'])[i], xi=list(df_filter['x_increment'])[i],
                                         wf=wf)
            try:
                df_wf[f'ch_{i}'] = wf
                df_wf[f'time_{i}'] = x_time
            except ValueError:
                print("The number of data points in the waveform doesn't match the length of the DataFrame, ")
                print(f"which is {first_wf_len} points.")
                pass
        print(wf_length_count)
        return df_wf

    @staticmethod
    def make_dir(save_name):
        """
        Creates directory if it doesn't exist
        :param save_name:
        """
        directory = '\\'.join(save_name.split('\\')[:-1])
        if not os.path.exists(directory):
            os.makedirs(directory)

    def plot_mask(self, df, v_max, v_min):
        """
        Given a DataFrame with columns 'volt' and 'time', v_max and v_min, return 3 DataFrames.
        upper includes data above the max, lower includes data below the max and middle for nominal data
        :param df:
        :param v_max: steady state max or max overshoot
        :param v_min: steady state min or min undershoot
        :return: 3 DataFrames
        """
        upper = df.loc[df['volt'] > v_max]
        lower = df.loc[df['volt'] < v_min]
        middle = df.loc[(df['volt'] >= v_min) & (df['volt'] <= v_max)]
        return upper, lower, middle

    def monotonicity(self, df_mono):
        """
        Given df_mono, determines monotonicity using percent change, which is a rolling calculation,
        chaning window_points will change the window size of the calculation
        :param df_mono:
        :return: m_color, m_annotation: strings: graph color and annotation
        """
        # check monotonicity
        print(df_mono.info())
        print(df_mono.head())
        print(df_mono.tail())

        df_length = len(df_mono)
        initial_time = df_mono['time'].iloc[0]
        final_time = df_mono['time'].iloc[-1]
        print('Initial Time: ', initial_time)
        print('Final Time: ', final_time)
        print('Rise Time: ', final_time - initial_time)
        print('DataFrame Length: ', df_length)

        time_increment = (df_mono['time'].iloc[-1] - df_mono['time'].iloc[0])/len(df_mono)
        window_time = 0.1 * time_increment

        window_points = int(0.3 * df_length)
        print('Window (time): ', window_time)
        print('Data points in window: ', window_points)
        pct_chg = df_mono['volt'].pct_change(periods=window_points).reset_index(drop=True)[window_points:]
        print(f'Percent Change Min: {pct_chg.min()} and Max: {pct_chg.max()}')
        monotonicity = np.all(pct_chg > 0)
        if monotonicity:
            m_color = 'b'
            m_annotation = 'monotonic'
        else:
            m_color = 'r'
            m_annotation = 'non-monotonic'

        return m_color, m_annotation

    def plot_calculations(self, df, tp):
        """
        Given the DataFrame, access testpoint parameters, perform calculations and produce
        3 new DataFrames representing the rising edge for monotonicity testing, ring measurement
        and steady state.  ss -> steady state, us -> undershoot
        :param df: DataFrame with volt and time column
        :param tp: testpoint being plotted
        :return df_mono, df_ring, df_ss: DataFrames
        :return df_ss, ss_max, ss_min, time_exp_settled, time_max, time_min, time_undershoot_min: floats
        :return m_annotation, m_color: strings
        """
        df = df.astype('float64')
        settle_time = self.plot_multi_params[tp]['settle_time']
        target_v = self.plot_multi_params[tp]['target_v']
        ss_max = self.plot_multi_params[tp]['steady_state_max']
        ss_min = self.plot_multi_params[tp]['steady_state_min']
        us_min = self.plot_multi_params[tp]['undershoot_min']
        v_10per = 0.1 * target_v
        v_90per = 0.9 * target_v

        time_undershoot_min = df['time'].loc[df['volt'] >= us_min].min()

        time_exp_settled = time_undershoot_min + settle_time
        time_max = df['time'].max()
        time_min = df['time'].min()

        # waveform slice filters
        df_mono = df.loc[(df['volt'] >= v_10per) &
                         (df['volt'] <= v_90per) &
                         (df['time'] <= time_undershoot_min)]
        df_ring = df.loc[(df['time'] >= time_undershoot_min) &
                         (df['time'] <= time_exp_settled)]
        df_ss = df.loc[(df['time'] >= time_exp_settled)]

        # check monotonicity
        m_color, m_annotation = self.monotonicity(df_mono)

        return df_mono, df_ring, df_ss, ss_max, ss_min, time_exp_settled, \
               time_max, time_min, m_annotation, m_color, time_undershoot_min

    def plot_multi(self, df_wf, save_name, i):
        """
        Splits waveform into monotonicity, ringing and steady state sections and adds those plots to the figure
        :param df_wf:
        :param save_name:
        :param i: waveform attributes ('23A0', '1P4V', 11.6, 3.3, 25, 1000)
        """
        tp = i[1]  # test point

        cnt = int(len(df_wf.columns) / 2)
        # Plot each channel in the figure
        for x in range(0, cnt):
            us_min = self.plot_multi_params[tp]['undershoot_min']
            df_pm = pd.DataFrame()
            df_pm['time'] = df_wf[f'time_{x}']
            df_pm['volt'] = df_wf[f'ch_{x}']
            df_pm = df_pm.astype('float64')
            v_wf_max = df_pm['volt'].max()

            if v_wf_max < us_min:
                plt.figure(figsize=(15, 10))
                warning = f'Warning: Maximum Waveform Voltage ({v_wf_max}) is Less than Undershoot Min ({us_min})'
                # full plot
                # plt.subplot(4, 1, 1)
                plt.plot(df_pm['time'], df_pm['volt'], c='r')

                plt.hlines(self.plot_multi_params[tp]['target_v'], df_pm.time.iloc[0],
                           df_pm.time.iloc[-1], label='Expected', colors='b')

                title_obj = plt.title(f'Full Range - {warning}')
                plt.setp(title_obj, color='r')
                plt.ylabel('Voltage (v)')
                plt.legend()
            else:
                plt.figure(figsize=(15, 20))
                calcs = self.plot_calculations(df_pm, tp)

                # full plot
                plt.subplot(4, 1, 1)
                plt.plot(df_pm['time'], df_pm['volt'])

                plt.hlines(self.plot_multi_params[tp]['target_v'], calcs[7], calcs[6], label='Expected', colors='b')
                plt.title('Full Range')
                plt.ylabel('Voltage (v)')
                plt.legend()

                # monotonicity plot
                plt.subplot(4, 1, 2)
                plt.plot(calcs[0]['time'], calcs[0]['volt'], c=calcs[9])
                plt.title(f'Rising Edge 10% - 90%: {calcs[8]}')
                plt.ylabel('Voltage (v)')
                plt.legend()

                # ring measurement
                os_max = self.plot_multi_params[tp]['overshoot_max']
                us_min = self.plot_multi_params[tp]['undershoot_min']
                plt.subplot(4, 1, 3)
                # plt.plot(calcs[1]['time'], calcs[1]['volt'])

                # add masking
                ss_mask = self.plot_mask(calcs[1], os_max, us_min)
                for i, level in enumerate(ss_mask):
                    if i == 2:
                        mask_color = 'b'
                    else:
                        mask_color = 'r'
                    plt.plot(level['time'], level['volt'], c=mask_color)

                # add horizontal lines
                plt.hlines(self.plot_multi_params[tp]['target_v'], calcs[10] - 0.001,
                           calcs[5] + 0.001, label='Expected', colors='b')
                plt.hlines(os_max, calcs[10] - 0.001, calcs[5] + 0.001, label='Max', colors='r')
                plt.hlines(us_min, calcs[10] - 0.001, calcs[5] + 0.001, label='Min', colors='r')
                plt.title('Ring Measurement')
                plt.ylabel('Voltage (v)')
                # plt.legend()

                # steady state
                plt.subplot(4, 1, 4)
                # plt.plot(calcs[2]['time'], calcs[2]['volt'])

                # add masking
                ss_mask = self.plot_mask(calcs[2], calcs[3], calcs[4])
                for i, level in enumerate(ss_mask):
                    if i == 2:
                        mask_color = 'b'
                    else:
                        mask_color = 'r'
                    plt.plot(level['time'], level['volt'], c=mask_color)

                # add horizontal lines
                plt.hlines(self.plot_multi_params[tp]['target_v'], calcs[5] - 0.001,
                           calcs[6] + 0.001, label='Expected', colors='b')
                plt.hlines(calcs[3], calcs[5] - 0.001, calcs[6] + 0.001, label='Max', colors='r')
                plt.hlines(calcs[4], calcs[5] - 0.001, calcs[6] + 0.001, label='Min', colors='r')
                plt.title('Steady State')
                plt.ylabel('Voltage (v)')
                plt.xlabel('time (s)')

        self.make_dir(save_name)
        plt.savefig(f'{save_name}.png')
        plt.close('all')
        del df_wf
        gc.collect()

    def plot(self, df_wf, save_name):
        # Plot Waveforms
        plt.figure(figsize=(15, 10))
        cnt = int(len(df_wf.columns) / 2)
        # Plot each channel in the figure
        for x in range(0, cnt):
            plt.plot(df_wf[f'time_{x}'], df_wf[f'ch_{x}'])
        y_scale = max(df_wf.max()) + 0.5
        if y_scale >= 6:
            step = 1
        elif 3 <= y_scale < 6:
            step = 0.5
        else:
            step = 0.25
        major_ticks = np.arange(0, y_scale, step)  # y_scale/12
        if len(df_wf.columns) / 2 < 10:
            plt.legend()
        plt.xlabel('Time(s)')
        plt.ylabel('Voltage(V)')

        plt.yticks(major_ticks)
        plt.grid()
        self.make_dir(save_name)
        plt.savefig(f'{save_name}.png')
        plt.close('all')
        del df_wf
        gc.collect()

    def iter_filter(self, test_tuple, params, dirs, create_df, plot_multi=False):
        """
        Using itertools.product, iterates through all combinations of test_tuple parameters.  Using the combinations,
        a filter is created and applied to the DataFrame
        :param test_tuple: class attribute values from each row in the DataFrame passed into the class
        :param params: list  of strings used to generate unique filters
        :param dirs: directory to save waveforms
        :param create_df: if True, creates Dict of DataFrames of waveforms
        :param plot_multi: Flag for using plot_multi if true otherwise just uses plot...only for single waveform graphs
        :return: if create_df is True, returns a dict of DataFrames (memory intensive)
        """
        df_dict = {}
        dirs = os.path.join(self.plot_dir, dirs)
        for i in itertools.product(*[b for _, b in test_tuple]):
            print(f'i: {i}')
            print('\n'.join(f'{a}:{b}' for a, b in zip(params, i)))  # keep - used for printing information only
            name_params = '_'.join(f'{b}-{a}' for a, b in zip(params, i))  # keep - this is used
            save_name = f'{dirs}\\{self.dut[1][0]}_{name_params}'.replace('.', 'p')
            print(f'Save Name: {save_name}')

            filter_t = ' & '.join(f'{c[0]} == "{b}"' for b, c in zip(i, test_tuple))
            print(f'filter_t: {filter_t}')
            filtered_df = self.df.loc[self.df.eval(filter_t)]

            df_wf = self.create_wf_df(filtered_df)
            if len(df_wf.columns) == 0:
                print('*' * 10, ' Filter Conditions Result in Empty DataFrame', '*' * 10)
            elif plot_multi:
                if create_df:
                    df_dict[name_params] = df_wf
                self.plot_multi(df_wf, save_name, i)
                del df_wf
                gc.collect()
            else:
                if create_df:
                    df_dict[name_params] = df_wf
                self.plot(df_wf, save_name)
                del df_wf
                gc.collect()
        print('Finished')
        if create_df:
            return df_dict

    def tp_slew(self, create_df=False):
        """
        Test point and slew rate are used to determine the groupings for these output waveforms
        :param create_df:
        :return df_dict:
        """
        params = ['tp', 'v1', 'slew']
        dirs = 'startup\\tp_slew'
        df_dict = self.iter_filter(self.tp_slew_list, params, dirs, create_df)
        return df_dict

    def tp_temp(self, create_df=False):
        """
        Test point and temperature are used to determine the groupings for these output waveforms
        :param create_df:
        :return df_dict:
        """
        params = ['tp', 'v1', 'temp']
        dirs = 'startup\\tp_temp'
        df_dict = self.iter_filter(self.tp_temp_list, params, dirs, create_df)
        return df_dict

    def seq_1ps(self, create_df=False):
        """
        Sequencing when there is one power supply (v1)
        :param create_df:
        :type: bool
        :return: df_dict
        :type: dict
        """
        params = ['slew', 'v1']
        dirs = 'startup\\sequencing'
        df_dict = self.iter_filter(self.seq_1ps_list, params, dirs, create_df)
        return df_dict

    def seq_2ps(self, create_df=False):
        """
        Sequencing when there are two power supplies (v1, v2), where v2 has more than 1 voltage
        :param create_df:
        :type: bool
        :return: df_dict
        :type: dict
        """
        params = ['slew', 'v1', 'v2']
        dirs = 'startup\\sequencing'
        df_dict = self.iter_filter(self.seq_2ps_list, params, dirs, create_df)
        return df_dict

    def five_test_conditions(self, create_df=False, plot_multi=False):
        """
        Used to plot single waveform graphs for 1 power supplies (if there are other power supplies, but only with one
        voltage, use this)
        :param create_df: True if you want a dict of all waveform DataFrames. 99 waveforms of 1.25M data points is 2GB
        :param plot_multi: True if you want 4 plots per figure rising edge, ringing, steady state and full
        :return: Dictionary of DataFrames
        """
        params = ['sn', 'tp', 'v1', 'temp', 'slew']
        dirs = 'startup\\five_tc_wf'
        df_dict = self.iter_filter(self.five_list, params, dirs, create_df, plot_multi)
        return df_dict

    def six_test_conditions(self, create_df=False, plot_multi=False):
        """
        Used to plot single waveform graphs for 2 power supplies (each with more than one voltage)
        :param create_df: True if you want a dict of all waveform DataFrames. 99 waveforms of 1.25M data points is 2GB
        :param plot_multi: True if you want 4 plots per figure rising edge, ringing, steady state and full
        :return: Dictionary of DataFrames
        """
        params = ['sn', 'tp', 'v1', 'v2', 'temp', 'slew']
        dirs = 'startup\\six_tc_wf'
        df_dict = self.iter_filter(self.six_list, params, dirs, create_df, plot_multi)
        return df_dict

    def seven_test_conditions(self, create_df=False, plot_multi=False):
        """
        Used to plot single waveform graphs for 3 power supplies (each with more than one voltage)
        :param create_df: True if you want a dict of all waveform DataFrames. 99 waveforms of 1.25M data points is 2GB
        :param plot_multi: True if you want 4 plots per figure rising edge, ringing, steady state and full
        :return: Dictionary of DataFrames
        """
        params = ['sn', 'tp', 'v1', 'v2', 'v3', 'temp', 'slew']
        dirs = 'startup\\seven_tc_wf'
        df_dict = self.iter_filter(self.seven_list, params, dirs, create_df, plot_multi)
        return df_dict


class ReadWaveform:
    def __init__(self, data_frame):
        self.df = data_frame

    @staticmethod
    def read_waveform(data_address, print_enable=False):
        """
        Uses numpy to create an array for the specified waveform
        :param print_enable:
        :param data_address: string -> path and file name
        :return: array, float
        """
        f = ZipFile(os.path.join(data_address), 'r')
        f_name = f.namelist()
        if print_enable:
            print(f_name[0])
        w = f.read(f_name[0])  # the key is the file name without the 'bin' extension (i.e. CH1)
        if print_enable:
            print(len(w))
        f.close()
        dt = np.dtype(float)
        dt = dt.newbyteorder('>')
        wf = np.frombuffer(w, dtype=dt)
        return wf

    def len_data_files(self, p_e=False):
        """
        Given a DataFrame:
        1. reads the address to the data
        2. sends the address to read_waveform
        3. the length of the waveform is determined and appended to the lengths list
        4. the list is saved as a DataFrame column
        :type p_e: print_enable to be passed to read_waveform
        """
        lengths = []
        for row in self.df['data_address']:
            wf = self.read_waveform(row, print_enable=p_e)
            lengths.append(len(wf))
        self.df['data_length'] = lengths
        return self.df
