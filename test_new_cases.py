from datetime import datetime

data_handler = DataHandler()

viz_start_date = datetime(2021, 1, 1)
viz_end_date = datetime(2021, 12, 31)

viz_start_date = pd.Timestamp(viz_start_date).tz_localize('America/New_York')
viz_end_date = pd.Timestamp(viz_end_date).tz_localize('America/New_York')
