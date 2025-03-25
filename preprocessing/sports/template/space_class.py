from .basketball.basketball_space_class import Basketball_space_data

class Space_data:
    # Modified the sports list to only include fully supported providers
    sports = ['provider_1', 'provider_2']

    def __new__(cls, data_provider, *args, **kwargs):
        if data_provider in cls.sports:
            # If the data_provider is in the supported list, return an instance of Soccer_SAR_data
            return Basketball_space_data(data_provider, *args, **kwargs)
        else:
            # If the data_provider is unrecognized, raise a ValueError
            raise ValueError(f'Unknown data provider: {data_provider}')

if __name__ == '__main__':
    # Test block remains unchanged, using a supported provider 
    provider_path = "./provider_1/Data/"
    match_id = "1234"
    Space_data(data_provider='provider_1', data_path = provider_path,...).load_data()
