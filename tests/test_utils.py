import pytest
import os
from pandas import Timestamp, DataFrame
from pandas.testing import assert_frame_equal
from geopandas import GeoDataFrame
from s2sphere import CellId
from shapely.geometry import LineString, Point
from pam import utils, plot, read
from pam.core import Household, Population
from tests.fixtures import instantiate_household_with

test_trips_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "test_data/test_matsim_plans.xml")
)


@pytest.fixture()
def pt_person():
    return read.read_matsim(test_trips_path)['census_1']['census_1']


@pytest.fixture()
def correct_pt_person_geodataframe():
    return GeoDataFrame({
                         '_distance': {0: 6412.774690, 1: 1290.851862,
                                          2: 5952.837241,
                                          3: 13.313250, 4: 12049.487855,
                                          5: 0.000000,
                                          6: 5731.316434, 7: 2606.715618},
                        'd_stop': {0: None,
                                    1: None,
                                    2: '9100UPMNSP6.link:302438',
                                    3: None,
                                    4: '9100BARKING.link:15983',
                                    5: None,
                                    6: '9100DGNHMDC.link:280957',
                                    7: None},
                         'end_location': {0: (550873.0, 187629.0), 1: (551390.0, 188476.0),
                                          2: (556131.0, 186870.0),
                                          3: (556131.0, 186860.0), 4: (544411.0, 184340.0),
                                          5: (544411.0, 184340.0),
                                          6: (548976.0, 182980.0), 7: (547360.0, 184166.0)},
                         'end_time': {0: Timestamp('1900-01-01 08:38:15'), 1: Timestamp('1900-01-01 20:55:49'),
                                      2: Timestamp('1900-01-01 21:19:59'), 3: Timestamp('1900-01-01 21:20:14'),
                                      4: Timestamp('1900-01-01 21:38:58'), 5: Timestamp('1900-01-01 21:38:58'),
                                      6: Timestamp('1900-01-01 21:43:58'), 7: Timestamp('1900-01-01 22:36:06')},
                         'freq': {0: None,
                                    1: None,
                                    2: None,
                                    3: None,
                                    4: None,
                                    5: None,
                                    6: None,
                                    7: None},
                         'geometry': {0: LineString([(547360.0, 184166.0), (550873.0, 187629.0)]),
                                      1: LineString([(550873.0, 187629.0), (551390.0, 188476.0)]),
                                      2: LineString([(551390.0, 188476.0), (556131.0, 186870.0)]),
                                      3: LineString([(556131.0, 186870.0), (556131.0, 186860.0)]),
                                      4: LineString([(556131.0, 186860.0), (544411.0, 184340.0)]),
                                      5: LineString([(544411.0, 184340.0), (544411.0, 184340.0)]),
                                      6: LineString([(544411.0, 184340.0), (548976.0, 182980.0)]),
                                      7: LineString([(548976.0, 182980.0), (547360.0, 184166.0)])},
                         'mode': {0: 'transit_walk', 1: 'transit_walk', 2: 'pt', 3: 'transit_walk', 4: 'pt',
                                  5: 'transit_walk', 6: 'pt', 7: 'transit_walk'},
                         'network_route': {0: None,
                                           1: None,
                                           2: None,
                                           3: None,
                                           4: None,
                                           5: None,
                                           6: None,
                                           7: None
                                           },
                         'o_stop': {0: None,
                                    1: None,
                                    2: '9100ROMFORD.link:25821',
                                    3: None,
                                    4: '9100UPMNSTR.link:84556',
                                    5: None,
                                    6: '9100BARKING.link:15984',
                                    7: None},
                         'purp': {0: 'work', 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None},
                         'route_id': {0: None,
                                      1: None,
                                      2: 'VJ307b99b535bf55bc9d62b5475e5edf0d37176bcf',
                                      3: None,
                                      4: 'VJd0adc2f9ee1d3687b3a36b84ff2a2e4c7b31f621',
                                      5: None,
                                      6: 'VJc808d717b38ec29c015695e5d4ff576e79af623f',
                                      7: None},
                         'seq': {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0, 6: 7.0, 7: 8.0},
                         'service_id': {0: None, 1: None, 2: '25239', 3: None, 4: '162', 5: None, 6: '723',
                                        7: None},
                         'start_location': {0: (547360.0, 184166.0), 1: (550873.0, 187629.0),
                                            2: (551390.0, 188476.0),
                                            3: (556131.0, 186870.0), 4: (556131.0, 186860.0),
                                            5: (544411.0, 184340.0),
                                            6: (544411.0, 184340.0), 7: (548976.0, 182980.0)},
                         'start_time': {0: Timestamp('1900-01-01 06:30:00'),
                                        1: Timestamp('1900-01-01 20:30:00'),
                                        2: Timestamp('1900-01-01 20:55:49'),
                                        3: Timestamp('1900-01-01 21:19:59'),
                                        4: Timestamp('1900-01-01 21:20:14'),
                                        5: Timestamp('1900-01-01 21:38:58'),
                                        6: Timestamp('1900-01-01 21:38:58'),
                                        7: Timestamp('1900-01-01 21:43:58')},
                         'pid': {0: 'census_1',
                                 1: 'census_1',
                                 2: 'census_1',
                                 3: 'census_1',
                                 4: 'census_1',
                                 5: 'census_1',
                                 6: 'census_1',
                                 7: 'census_1'}})


@pytest.fixture()
def cyclist():
    return read.read_matsim(test_trips_path)['census_2']['census_2']


@pytest.fixture()
def correct_cyclist_geodataframe():
    return GeoDataFrame(
        {'_distance': {0: 9647.24116945626, 1: 9647.241169456261},
        'd_stop': {0: None, 1: None},
         'end_location': {0: (529652.0, 183897.0), 1: (515226.0, 188222.0)},
         'end_time': {0: Timestamp('1900-01-01 08:07:08'), 1: Timestamp('1900-01-01 15:33:08')},
         'freq': {0: None, 1: None},
         'geometry': {0: LineString([(515226.0, 188222.0), (529652.0, 183897.0)]),
                      1: LineString([(529652.0, 183897.0), (515226.0, 188222.0)])},
         'mode': {0: 'bike', 1: 'bike'},
         'network_route': {0: ['link_1', 'link_2', 'link_3'], 1: ['link_3', 'link_2', 'link_1']},
         'o_stop': {0: None, 1: None}, 'purp': {0: 'work', 1: None},
         'route_id': {0: None, 1: None},
         'seq': {0: 1.0, 1: 2.0},
         'service_id': {0: None, 1: None},
         'start_location': {0: (515226.0, 188222.0), 1: (529652.0, 183897.0)},
         'start_time': {0: Timestamp('1900-01-01 07:24:52'), 1: Timestamp('1900-01-01 14:50:52')},
         'pid': {0: 'census_2', 1: 'census_2'}})


def test_build_geodataframe_for_pt_person(pt_person, correct_pt_person_geodataframe):
    gdf = pt_person.build_travel_geodataframe()
    gdf = gdf[correct_pt_person_geodataframe.columns]
    assert_frame_equal(gdf, correct_pt_person_geodataframe)


def test_building_geodataframe_with_reprojection(pt_person, correct_pt_person_geodataframe):
    gdf = pt_person.build_travel_geodataframe(from_epsg='epsg:27700', to_epsg='epsg:4326')
    assert [[(round(p[0], 6), round(p[1], 6)) for p in ls.coords] for ls in gdf['geometry'].to_list()] == [
        [(0.123336, 51.537168), (0.175434, 51.567355)], [(0.175434, 51.567355), (0.183252, 51.574827)],
        [(0.183252, 51.574827), (0.250898, 51.559106)], [(0.250898, 51.559106), (0.250894, 51.559016)],
        [(0.250894, 51.559016), (0.080917, 51.539493)], [(0.080917, 51.539493), (0.080917, 51.539493)],
        [(0.080917, 51.539493), (0.146117, 51.526088)], [(0.146117, 51.526088), (0.123336, 51.537168)]]


def test_build_geodataframe_for_cyclist(cyclist, correct_cyclist_geodataframe):
    gdf = cyclist.build_travel_geodataframe()
    gdf = gdf[correct_cyclist_geodataframe.columns]
    assert_frame_equal(gdf, correct_cyclist_geodataframe)


def test_build_hhld_geodataframe(pt_person, cyclist, correct_pt_person_geodataframe, correct_cyclist_geodataframe):
    hhld = instantiate_household_with([pt_person, cyclist], hid='household_id')
    gdf = hhld.build_travel_geodataframe()
    assert 'hid' in gdf.columns

    correct_gdf = correct_pt_person_geodataframe
    correct_gdf = correct_gdf.append(correct_cyclist_geodataframe)
    correct_gdf = correct_gdf.reset_index(drop=True)
    correct_gdf['hid'] = 'household_id'
    gdf = gdf[correct_gdf.columns]
    assert_frame_equal(gdf, correct_gdf)


def test_build_pop_geodataframe(pt_person, cyclist, correct_pt_person_geodataframe, correct_cyclist_geodataframe):
    pop = Population()
    pop.add(instantiate_household_with([pt_person, cyclist], hid='1'))
    pop.add(instantiate_household_with([pt_person], hid='2'))
    pop.add(instantiate_household_with([cyclist], hid='3'))
    gdf = pop.build_travel_geodataframe()
    assert 'hid' in gdf.columns

    correct_gdf = correct_pt_person_geodataframe
    correct_gdf = correct_gdf.append(correct_cyclist_geodataframe)
    correct_gdf['hid'] = '1'

    correct_pt_person_geodataframe['hid'] = '2'
    correct_gdf = correct_gdf.append(correct_pt_person_geodataframe)

    correct_cyclist_geodataframe['hid'] = '3'
    correct_gdf = correct_gdf.append(correct_cyclist_geodataframe)

    correct_gdf = correct_gdf.reset_index(drop=True)
    gdf = gdf[correct_gdf.columns]
    assert_frame_equal(gdf, correct_gdf)


def test_get_linestring_with_s2_cellids():
    from_point = CellId(5221390681063996525)
    to_point = CellId(5221390693823388667)

    ls = utils.get_linestring(from_point, to_point)
    assert isinstance(ls, LineString)
    assert [(round(c[0], 6), round(c[1], 6)) for c in ls.coords] == [(-0.137925, 51.521699), (-0.134456, 51.520027)]
