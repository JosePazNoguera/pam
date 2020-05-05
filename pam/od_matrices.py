import pandas as pd


def extract_od(population, path):
    ozone = []
    dzone = []

    for hid, household in population.households.items():
        for pid, person in household.people.items():
            for p in person.plan:
                if p.act == 'travel':
                    o = p.start_location.area
                    d = p.end_location.area
                    ozone.append(o)
                    dzone.append(d)

    data_dict = {
        'ozone': ozone,
        'dzone': dzone
    }

    df = pd.DataFrame(data=data_dict).set_index('ozone')
    matrix = df.pivot_table(values='dzone', index='ozone', columns='dzone', fill_value=0, aggfunc=len)

    matrix.to_csv(path)