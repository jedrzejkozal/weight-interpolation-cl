import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from plot_backup import get_accs_bic, get_accs_derpp, get_accs_er_ace

def get_accs():
    "no alpha schedule"
    return [
        [[49.5, 90.10000000000001], [53.1, 90.0], [56.2, 89.7], [59.4, 89.0], [61.9, 88.1], [64.8, 87.3], [67.80000000000001, 87.3], [71.39999999999999, 86.2], [73.4, 84.7], [76.6, 82.39999999999999], [79.5, 80.7], [80.7, 78.10000000000001], [82.19999999999999, 75.1], [83.2, 70.8], [84.7, 65.5], [85.1, 59.599999999999994], [85.5, 52.800000000000004], [85.7, 45.1], [85.6, 38.4], [85.8, 30.599999999999998], [86.0, 24.099999999999998], [85.8, 15.9], [85.9, 10.7], [85.9, 6.1], [85.8, 3.6999999999999997], [85.7, 2.0], [85.6, 0.8999999999999999], [85.5, 0.6], [85.39999999999999, 0.3], [85.39999999999999, 0.0], [85.39999999999999, 0.0], [85.5, 0.0], [85.3, 0.0], [85.3, 0.0], [85.7, 0.0], [85.6, 0.0], [85.8, 0.0], [85.5, 0.0], [85.3, 0.0], [84.89999999999999, 0.0], [85.0, 0.0], [85.2, 0.0], [85.39999999999999, 0.0], [85.0, 0.0], [85.39999999999999, 0.0], [85.3, 0.0], [85.5, 0.0], [85.0, 0.0], [84.89999999999999, 0.0], [84.8, 0.0], [84.39999999999999, 0.0]],
        [[32.300000000000004, 32.1, 87.5], [35.5, 34.8, 87.1], [39.1, 37.9, 86.8], [41.9, 39.900000000000006, 86.6], [45.2, 43.6, 85.8], [48.699999999999996, 46.9, 85.3], [51.9, 49.9, 84.2], [54.6, 53.6, 82.69999999999999], [57.4, 57.099999999999994, 81.69999999999999], [60.0, 61.0, 79.7], [62.1, 63.6, 77.60000000000001], [64.60000000000001, 66.3, 74.8], [66.60000000000001, 69.0, 72.6], [67.9, 71.6, 67.9], [68.5, 74.0, 63.6], [70.0, 75.9, 59.9], [71.0, 77.9, 54.50000000000001], [72.0, 78.4, 48.4], [72.7, 79.80000000000001, 42.199999999999996], [73.2, 80.5, 35.699999999999996], [74.0, 81.2, 28.000000000000004], [75.0, 81.5, 20.1], [74.7, 82.19999999999999, 13.200000000000001], [74.7, 82.8, 8.4], [75.2, 83.0, 5.0], [75.4, 83.0, 3.0], [75.2, 82.6, 1.4000000000000001], [75.4, 83.0, 0.5], [75.1, 82.69999999999999, 0.1], [74.8, 82.8, 0.1], [75.4, 83.1, 0.1], [75.9, 83.2, 0.1], [76.1, 82.89999999999999, 0.0], [76.1, 82.8, 0.0], [76.6, 82.69999999999999, 0.0], [76.4, 82.6, 0.0], [76.5, 82.6, 0.0], [76.3, 82.1, 0.0], [76.7, 81.69999999999999, 0.0], [77.0, 81.5, 0.0], [76.8, 81.6, 0.0], [76.8, 81.69999999999999, 0.0], [76.9, 81.39999999999999, 0.0], [76.8, 81.0, 0.0], [76.9, 80.7, 0.0], [77.0, 80.60000000000001, 0.0], [76.9, 80.0, 0.0], [77.10000000000001, 79.60000000000001, 0.0], [76.9, 79.3, 0.0], [76.8, 79.0, 0.0], [76.7, 78.9, 0.0]],
        [[29.5, 29.099999999999998, 37.1, 90.60000000000001], [30.9, 31.0, 39.300000000000004, 90.60000000000001], [32.9, 33.6, 41.5, 90.4], [35.699999999999996, 36.6, 44.0, 90.0], [36.9, 38.3, 46.5, 89.5], [38.9, 40.0, 48.6, 88.9], [40.8, 42.8, 50.3, 88.5], [43.0, 45.0, 52.6, 87.6], [45.5, 46.9, 55.900000000000006, 86.5], [47.3, 48.8, 58.3, 85.0], [49.5, 50.8, 61.1, 82.8], [51.1, 52.7, 63.4, 80.2], [53.300000000000004, 55.00000000000001, 65.4, 77.9], [55.2, 56.39999999999999, 67.0, 74.6], [56.99999999999999, 58.4, 68.4, 69.89999999999999], [58.699999999999996, 58.8, 69.69999999999999, 65.5], [60.0, 59.699999999999996, 71.3, 60.6], [60.9, 60.9, 72.7, 53.800000000000004], [61.7, 62.1, 74.1, 46.800000000000004], [62.4, 62.2, 74.6, 39.6], [63.5, 62.4, 75.1, 32.6], [63.7, 62.6, 75.3, 24.0], [63.800000000000004, 62.9, 75.5, 17.1], [63.6, 63.1, 75.2, 11.4], [63.6, 63.4, 75.2, 6.4], [63.6, 63.800000000000004, 75.3, 2.9000000000000004], [63.4, 63.6, 75.4, 1.2], [63.4, 63.800000000000004, 75.6, 0.4], [63.1, 63.2, 76.0, 0.2], [63.0, 62.9, 75.9, 0.0], [63.0, 62.9, 75.9, 0.0], [62.8, 62.7, 75.8, 0.0], [62.8, 62.5, 76.0, 0.0], [63.0, 62.3, 76.2, 0.0], [62.6, 62.4, 76.1, 0.0], [62.5, 62.1, 76.2, 0.0], [62.3, 62.3, 76.0, 0.0], [62.0, 62.5, 75.8, 0.0], [61.7, 62.6, 75.6, 0.0], [61.1, 62.5, 75.4, 0.0], [60.3, 62.3, 75.1, 0.0], [60.0, 62.5, 75.2, 0.0], [60.099999999999994, 62.5, 74.9, 0.0], [60.099999999999994, 62.1, 74.8, 0.0], [60.099999999999994, 61.3, 75.0, 0.0], [59.599999999999994, 60.5, 74.9, 0.0], [59.3, 60.4, 74.3, 0.0], [58.699999999999996, 60.3, 73.6, 0.0], [58.3, 60.099999999999994, 73.3, 0.0], [57.699999999999996, 59.8, 73.1, 0.0], [56.89999999999999, 59.599999999999994, 72.6, 0.0]],
        [[21.4, 18.9, 30.3, 37.7, 84.1], [23.5, 21.4, 31.900000000000002, 39.900000000000006, 84.0], [25.5, 23.7, 33.7, 42.3, 83.8], [27.700000000000003, 25.3, 35.5, 44.6, 83.39999999999999], [28.799999999999997, 26.3, 38.1, 47.199999999999996, 83.1], [30.7, 28.199999999999996, 40.400000000000006, 48.8, 82.69999999999999], [31.7, 30.3, 42.3, 51.2, 82.19999999999999], [33.1, 32.1, 43.8, 54.1, 80.9], [35.4, 34.0, 45.4, 56.599999999999994, 80.2], [37.2, 35.4, 47.599999999999994, 58.5, 79.4], [39.6, 37.0, 49.0, 60.9, 78.0], [41.6, 38.4, 51.6, 63.0, 76.2], [43.0, 40.400000000000006, 53.2, 65.0, 74.2], [44.6, 42.3, 54.900000000000006, 65.7, 71.2], [45.6, 44.3, 56.2, 67.0, 68.7], [47.099999999999994, 45.6, 57.4, 68.4, 64.9], [47.699999999999996, 46.7, 58.099999999999994, 69.69999999999999, 60.199999999999996], [48.1, 48.4, 58.599999999999994, 71.6, 54.50000000000001], [48.699999999999996, 50.1, 59.5, 72.6, 49.9], [49.9, 50.7, 60.699999999999996, 73.7, 44.6], [50.6, 51.4, 61.5, 74.2, 37.3], [51.1, 52.300000000000004, 62.2, 74.8, 30.4], [51.5, 53.5, 62.5, 75.1, 23.7], [51.4, 53.6, 62.5, 76.1, 17.1], [51.7, 53.800000000000004, 62.4, 76.6, 12.1], [52.0, 54.400000000000006, 62.5, 76.8, 6.9], [52.0, 54.50000000000001, 62.7, 77.2, 3.5999999999999996], [51.9, 54.7, 62.8, 77.60000000000001, 1.6], [51.800000000000004, 54.800000000000004, 63.1, 77.9, 0.8999999999999999], [52.2, 54.400000000000006, 62.8, 78.2, 0.4], [52.1, 54.50000000000001, 62.8, 77.9, 0.1], [51.800000000000004, 54.50000000000001, 62.5, 78.0, 0.1], [51.6, 54.50000000000001, 62.5, 78.2, 0.0], [51.4, 54.1, 62.7, 78.60000000000001, 0.0], [51.4, 53.800000000000004, 63.0, 78.60000000000001, 0.0], [51.6, 53.400000000000006, 62.7, 78.9, 0.0], [51.7, 53.1, 63.0, 78.60000000000001, 0.0], [51.300000000000004, 53.1, 62.6, 78.5, 0.0], [51.4, 53.5, 62.5, 78.2, 0.0], [50.7, 53.400000000000006, 62.4, 78.4, 0.0], [50.4, 53.400000000000006, 62.6, 78.5, 0.0], [50.0, 53.300000000000004, 62.5, 78.5, 0.0], [49.7, 53.400000000000006, 62.0, 78.5, 0.0], [49.3, 52.900000000000006, 61.7, 78.5, 0.0], [48.699999999999996, 52.7, 61.1, 78.7, 0.0], [48.6, 52.400000000000006, 61.1, 78.60000000000001, 0.0], [48.699999999999996, 51.4, 60.4, 78.9, 0.0], [47.9, 51.5, 60.699999999999996, 79.10000000000001, 0.0], [47.9, 51.2, 60.099999999999994, 79.0, 0.0], [47.5, 50.2, 59.099999999999994, 78.8, 0.0], [47.199999999999996, 49.7, 58.4, 79.3, 0.0]],
        [[18.8, 17.5, 24.7, 29.599999999999998, 18.8, 92.7], [20.5, 18.3, 26.900000000000002, 32.2, 21.9, 92.7], [21.9, 19.5, 28.599999999999998, 34.300000000000004, 23.7, 92.5], [24.3, 20.9, 30.4, 36.4, 26.5, 92.5], [25.5, 22.3, 31.7, 39.2, 29.9, 92.30000000000001], [27.400000000000002, 24.5, 33.2, 42.5, 32.6, 92.10000000000001], [28.499999999999996, 26.700000000000003, 35.9, 44.800000000000004, 35.199999999999996, 90.7], [29.5, 28.7, 37.8, 47.0, 38.6, 90.10000000000001], [31.4, 30.3, 39.2, 49.0, 41.199999999999996, 89.5], [33.300000000000004, 31.4, 41.5, 51.2, 43.7, 87.4], [34.599999999999994, 33.300000000000004, 42.1, 53.5, 47.4, 85.9], [35.9, 34.1, 44.1, 54.6, 50.4, 83.7], [37.0, 35.699999999999996, 45.800000000000004, 56.10000000000001, 52.800000000000004, 81.69999999999999], [38.3, 36.1, 46.800000000000004, 56.99999999999999, 55.00000000000001, 79.2], [38.4, 37.7, 48.6, 58.099999999999994, 57.99999999999999, 74.8], [38.9, 38.800000000000004, 49.6, 58.8, 61.5, 70.6], [38.9, 39.900000000000006, 50.3, 59.8, 64.1, 65.5], [39.5, 40.2, 51.2, 60.3, 66.0, 58.199999999999996], [40.400000000000006, 40.5, 51.5, 61.1, 68.4, 51.1], [40.2, 41.099999999999994, 51.6, 61.5, 69.39999999999999, 45.6], [40.300000000000004, 41.199999999999996, 51.7, 61.7, 70.39999999999999, 36.6], [40.5, 41.4, 52.5, 62.1, 71.0, 26.400000000000002], [40.5, 41.5, 52.6, 62.0, 71.7, 20.0], [40.6, 41.5, 52.6, 62.1, 71.89999999999999, 12.5], [40.6, 41.4, 52.7, 62.3, 72.5, 8.7], [40.699999999999996, 41.4, 52.7, 62.0, 72.8, 4.9], [40.9, 41.0, 52.5, 62.1, 73.4, 2.4], [40.5, 40.8, 52.400000000000006, 62.1, 74.0, 0.8999999999999999], [40.400000000000006, 40.8, 52.1, 61.7, 74.2, 0.3], [40.400000000000006, 40.6, 51.9, 61.7, 74.3, 0.2], [40.2, 40.400000000000006, 51.9, 61.3, 74.2, 0.0], [39.800000000000004, 40.300000000000004, 51.800000000000004, 61.199999999999996, 74.5, 0.0], [39.6, 39.900000000000006, 51.4, 60.8, 74.5, 0.0], [39.300000000000004, 40.0, 51.2, 60.5, 73.8, 0.0], [39.2, 39.4, 51.1, 59.8, 74.1, 0.0], [39.300000000000004, 39.300000000000004, 51.0, 60.0, 74.2, 0.0], [39.4, 39.4, 50.9, 59.9, 74.8, 0.0], [39.1, 38.9, 50.2, 59.199999999999996, 75.4, 0.0], [39.300000000000004, 38.800000000000004, 49.7, 59.199999999999996, 75.7, 0.0], [39.1, 38.2, 49.4, 58.4, 75.7, 0.0], [38.9, 38.1, 49.3, 57.8, 76.1, 0.0], [38.3, 37.5, 48.699999999999996, 56.8, 75.9, 0.0], [38.2, 37.6, 48.1, 56.3, 76.0, 0.0], [38.1, 37.4, 47.699999999999996, 55.2, 76.2, 0.0], [37.6, 37.5, 47.3, 54.50000000000001, 76.1, 0.0], [37.6, 37.1, 47.099999999999994, 54.0, 75.9, 0.0], [37.3, 36.8, 46.300000000000004, 53.5, 75.7, 0.0], [36.7, 36.6, 46.300000000000004, 52.7, 75.7, 0.0], [36.199999999999996, 36.1, 45.9, 52.2, 75.5, 0.0], [35.699999999999996, 35.9, 45.2, 51.2, 75.5, 0.0], [34.9, 35.4, 44.7, 50.4, 75.6, 0.0]],
        [[18.2, 13.4, 22.1, 23.1, 17.599999999999998, 28.1, 88.7], [18.8, 14.499999999999998, 23.400000000000002, 25.2, 18.6, 32.0, 88.6], [20.7, 15.5, 24.3, 27.0, 20.599999999999998, 35.199999999999996, 88.6], [21.5, 16.400000000000002, 25.7, 29.4, 21.9, 38.3, 88.7], [23.5, 17.299999999999997, 26.6, 31.6, 23.1, 41.5, 88.4], [24.8, 18.6, 27.6, 32.6, 25.0, 44.9, 87.9], [25.6, 20.3, 29.9, 35.0, 26.8, 48.5, 87.3], [26.700000000000003, 21.2, 31.6, 36.4, 28.7, 51.2, 86.9], [27.700000000000003, 22.8, 33.0, 38.9, 30.099999999999998, 53.400000000000006, 86.3], [28.4, 24.0, 34.5, 41.0, 32.4, 56.89999999999999, 85.1], [29.5, 24.8, 36.4, 42.4, 34.9, 60.099999999999994, 84.39999999999999], [30.4, 26.1, 38.0, 44.1, 36.7, 62.3, 82.0], [30.8, 27.3, 39.1, 45.300000000000004, 38.3, 66.10000000000001, 79.4], [31.5, 28.499999999999996, 39.900000000000006, 47.0, 39.7, 69.0, 76.4], [31.8, 29.7, 40.699999999999996, 48.0, 40.9, 70.7, 73.0], [32.5, 30.2, 42.0, 49.5, 42.3, 72.8, 69.3], [33.1, 31.3, 42.8, 50.7, 43.9, 74.2, 64.1], [33.4, 31.8, 43.2, 52.0, 45.0, 75.5, 59.599999999999994], [33.7, 32.4, 44.2, 53.1, 46.7, 76.8, 51.4], [34.0, 33.0, 44.3, 53.7, 47.599999999999994, 77.9, 42.6], [34.4, 33.4, 44.6, 54.7, 47.8, 79.10000000000001, 34.4], [35.0, 33.4, 44.7, 55.1, 48.3, 79.60000000000001, 27.1], [35.0, 33.4, 44.9, 54.900000000000006, 48.5, 80.60000000000001, 20.9], [35.099999999999994, 33.6, 45.2, 55.300000000000004, 48.5, 81.10000000000001, 13.700000000000001], [35.0, 33.7, 45.300000000000004, 55.60000000000001, 48.9, 81.3, 8.200000000000001], [35.099999999999994, 33.5, 45.4, 55.300000000000004, 48.4, 81.6, 5.0], [35.0, 33.7, 45.2, 54.900000000000006, 48.199999999999996, 82.3, 3.2], [35.099999999999994, 33.800000000000004, 45.1, 54.6, 48.3, 82.6, 1.5], [35.4, 33.6, 45.300000000000004, 54.6, 48.3, 82.8, 0.6], [35.3, 33.6, 45.300000000000004, 54.400000000000006, 48.6, 83.0, 0.1], [35.5, 33.6, 45.2, 54.1, 48.5, 83.2, 0.0], [35.0, 33.2, 45.0, 53.800000000000004, 48.5, 83.1, 0.0], [34.8, 33.4, 44.7, 53.5, 48.6, 83.39999999999999, 0.0], [34.599999999999994, 33.1, 44.3, 53.0, 48.6, 83.39999999999999, 0.0], [35.0, 33.0, 44.3, 52.7, 48.8, 83.7, 0.0], [34.2, 32.9, 44.0, 52.5, 48.4, 83.8, 0.0], [33.900000000000006, 33.0, 44.0, 52.300000000000004, 48.0, 84.2, 0.0], [33.900000000000006, 33.0, 44.0, 51.9, 48.1, 84.2, 0.0], [33.5, 33.2, 43.8, 52.2, 47.5, 84.7, 0.0], [32.9, 33.0, 43.6, 51.800000000000004, 47.3, 84.7, 0.0], [32.800000000000004, 32.800000000000004, 43.4, 51.9, 47.3, 84.89999999999999, 0.0], [32.6, 33.0, 43.0, 51.7, 46.7, 85.1, 0.0], [32.1, 32.7, 42.3, 51.300000000000004, 46.300000000000004, 85.2, 0.0], [32.300000000000004, 32.4, 42.1, 51.0, 45.800000000000004, 85.3, 0.0], [32.2, 32.300000000000004, 41.4, 50.6, 45.7, 85.3, 0.0], [32.1, 32.1, 40.9, 49.8, 45.7, 85.8, 0.0], [32.300000000000004, 32.0, 40.2, 49.1, 45.6, 85.8, 0.0], [32.1, 32.1, 39.800000000000004, 48.9, 45.2, 85.9, 0.0], [31.6, 31.4, 38.5, 48.3, 45.0, 86.3, 0.0], [31.5, 31.4, 37.8, 48.1, 44.7, 86.4, 0.0], [31.1, 31.3, 37.1, 47.599999999999994, 44.3, 86.1, 0.0]],
        [[10.100000000000001, 9.8, 14.399999999999999, 23.5, 13.4, 20.200000000000003, 25.4, 90.7], [10.8, 10.9, 15.4, 25.6, 15.299999999999999, 21.7, 27.800000000000004, 90.60000000000001], [11.700000000000001, 12.1, 16.6, 27.0, 17.299999999999997, 24.099999999999998, 31.4, 90.7], [12.5, 13.3, 18.3, 28.000000000000004, 18.6, 26.400000000000002, 34.4, 90.60000000000001], [13.900000000000002, 14.399999999999999, 19.6, 29.4, 20.200000000000003, 29.099999999999998, 38.0, 90.2], [15.4, 15.2, 20.8, 31.2, 21.6, 31.3, 40.699999999999996, 89.60000000000001], [16.400000000000002, 16.400000000000002, 21.5, 33.0, 23.799999999999997, 34.0, 43.7, 89.5], [18.0, 17.5, 23.5, 35.099999999999994, 25.3, 36.3, 46.7, 88.3], [19.400000000000002, 19.6, 24.7, 36.199999999999996, 27.3, 38.800000000000004, 49.9, 87.3], [20.599999999999998, 20.5, 26.0, 37.4, 28.599999999999998, 41.099999999999994, 53.300000000000004, 85.8], [22.0, 22.2, 28.1, 38.3, 30.9, 43.2, 56.00000000000001, 84.8], [23.7, 23.1, 30.2, 39.800000000000004, 32.7, 45.4, 59.599999999999994, 83.1], [24.9, 24.5, 31.5, 41.199999999999996, 33.7, 48.199999999999996, 62.7, 81.39999999999999], [26.400000000000002, 25.5, 32.7, 42.3, 34.9, 50.5, 64.60000000000001, 78.8], [27.3, 26.3, 33.7, 44.2, 35.6, 52.400000000000006, 66.60000000000001, 75.6], [28.799999999999997, 26.8, 35.0, 44.9, 35.9, 53.7, 68.8, 70.8], [29.799999999999997, 27.3, 36.4, 45.4, 37.1, 56.00000000000001, 70.6, 65.60000000000001], [30.3, 28.199999999999996, 37.3, 45.6, 38.0, 57.49999999999999, 72.1, 59.5], [31.2, 28.4, 38.0, 46.1, 39.300000000000004, 58.699999999999996, 73.2, 53.1], [31.8, 28.599999999999998, 38.5, 46.400000000000006, 39.300000000000004, 59.4, 74.2, 45.6], [31.8, 28.999999999999996, 38.6, 46.6, 39.900000000000006, 60.199999999999996, 75.2, 37.7], [32.0, 29.2, 38.9, 47.3, 40.300000000000004, 60.4, 75.6, 30.7], [31.8, 29.599999999999998, 39.6, 47.199999999999996, 40.699999999999996, 60.8, 76.4, 22.900000000000002], [32.1, 30.0, 39.4, 47.0, 40.8, 60.8, 77.5, 16.5], [31.8, 30.099999999999998, 39.300000000000004, 46.7, 40.400000000000006, 61.199999999999996, 77.7, 10.9], [31.900000000000002, 30.599999999999998, 39.300000000000004, 46.5, 40.300000000000004, 61.7, 77.9, 6.0], [31.4, 30.599999999999998, 39.300000000000004, 46.6, 40.300000000000004, 61.9, 77.8, 3.5999999999999996], [31.2, 30.7, 39.5, 46.5, 40.699999999999996, 62.1, 78.10000000000001, 1.5], [31.0, 30.4, 39.4, 46.300000000000004, 40.400000000000006, 62.3, 78.8, 0.8999999999999999], [30.7, 30.4, 39.2, 46.2, 40.5, 62.3, 79.10000000000001, 0.3], [30.599999999999998, 30.5, 39.4, 45.800000000000004, 39.7, 62.1, 79.60000000000001, 0.1], [30.599999999999998, 30.4, 39.4, 45.300000000000004, 39.6, 62.2, 79.7, 0.0], [30.3, 30.2, 39.0, 45.1, 39.2, 62.2, 79.7, 0.0], [30.2, 30.3, 38.7, 45.0, 38.7, 62.3, 80.30000000000001, 0.0], [30.0, 30.3, 38.2, 44.6, 38.5, 61.9, 81.0, 0.0], [29.7, 29.799999999999997, 38.4, 44.3, 37.9, 61.9, 81.10000000000001, 0.0], [29.5, 29.2, 37.7, 43.5, 37.2, 62.1, 81.10000000000001, 0.0], [29.2, 28.999999999999996, 37.3, 43.6, 37.4, 62.1, 81.5, 0.0], [29.4, 28.9, 37.2, 43.5, 37.1, 61.8, 81.39999999999999, 0.0], [29.099999999999998, 28.7, 37.1, 43.1, 37.0, 61.4, 81.5, 0.0], [28.9, 28.499999999999996, 36.3, 42.8, 36.8, 61.6, 81.69999999999999, 0.0], [28.9, 28.1, 35.8, 42.6, 36.5, 61.4, 81.69999999999999, 0.0], [28.199999999999996, 27.6, 35.8, 42.0, 35.5, 60.8, 81.69999999999999, 0.0], [28.000000000000004, 27.1, 35.699999999999996, 41.9, 34.9, 60.6, 81.8, 0.0], [27.700000000000003, 26.8, 35.5, 41.5, 34.300000000000004, 60.099999999999994, 81.89999999999999, 0.0], [27.3, 26.5, 34.8, 41.0, 33.7, 59.599999999999994, 82.39999999999999, 0.0], [27.0, 26.3, 34.2, 40.5, 33.7, 59.4, 82.39999999999999, 0.0], [26.200000000000003, 25.6, 33.5, 40.0, 33.300000000000004, 59.4, 82.39999999999999, 0.0], [25.6, 25.8, 33.0, 39.6, 32.5, 58.9, 82.5, 0.0], [25.6, 25.2, 32.9, 39.2, 31.6, 58.5, 82.0, 0.0], [25.2, 25.1, 32.300000000000004, 38.800000000000004, 31.3, 58.4, 82.3, 0.0]],
        [[10.0, 10.5, 13.5, 14.499999999999998, 13.5, 14.7, 15.299999999999999, 20.200000000000003, 90.5], [11.0, 11.5, 14.7, 15.9, 14.7, 15.7, 17.2, 22.6, 90.4], [12.0, 12.6, 16.1, 17.8, 15.5, 17.4, 19.3, 25.2, 90.3], [12.8, 13.700000000000001, 17.1, 19.5, 16.5, 19.0, 21.0, 27.800000000000004, 90.2], [13.5, 14.499999999999998, 18.3, 21.5, 18.6, 21.2, 22.6, 31.2, 90.2], [14.499999999999998, 15.8, 19.900000000000002, 22.8, 19.5, 22.900000000000002, 24.4, 34.300000000000004, 90.0], [15.6, 17.5, 21.099999999999998, 25.1, 20.5, 25.0, 27.200000000000003, 38.7, 89.3], [16.8, 18.3, 23.1, 26.6, 21.6, 26.900000000000002, 29.2, 41.0, 89.3], [17.599999999999998, 19.1, 24.0, 29.2, 22.8, 28.7, 32.800000000000004, 45.0, 88.7], [18.5, 20.8, 25.1, 30.3, 24.0, 31.3, 35.5, 49.2, 87.3], [19.8, 21.8, 25.900000000000002, 32.4, 24.7, 32.9, 37.4, 51.9, 85.9], [20.7, 23.0, 27.0, 33.6, 25.900000000000002, 36.4, 40.1, 55.800000000000004, 83.39999999999999], [21.9, 23.5, 28.499999999999996, 35.4, 27.1, 39.2, 42.199999999999996, 59.099999999999994, 80.60000000000001], [22.8, 24.7, 29.5, 36.8, 27.800000000000004, 42.199999999999996, 44.9, 62.1, 77.5], [23.5, 25.8, 30.2, 38.1, 28.999999999999996, 43.9, 47.3, 65.4, 73.9], [24.0, 26.700000000000003, 31.4, 39.7, 30.0, 45.5, 50.6, 68.2, 69.5], [24.6, 27.1, 32.4, 41.099999999999994, 30.8, 47.099999999999994, 52.300000000000004, 70.8, 63.0], [25.5, 28.1, 32.7, 41.9, 31.6, 48.1, 54.0, 72.2, 57.9], [25.900000000000002, 28.9, 33.0, 42.6, 32.7, 48.8, 54.800000000000004, 73.7, 51.1], [25.6, 29.299999999999997, 33.900000000000006, 42.9, 33.4, 48.9, 56.00000000000001, 75.1, 44.3], [26.1, 29.099999999999998, 34.300000000000004, 43.3, 33.4, 49.5, 56.599999999999994, 76.1, 36.1], [26.5, 28.999999999999996, 34.599999999999994, 43.2, 33.800000000000004, 49.8, 56.699999999999996, 76.9, 28.299999999999997], [26.400000000000002, 28.599999999999998, 34.4, 43.4, 34.300000000000004, 49.8, 57.099999999999994, 77.60000000000001, 19.0], [26.5, 28.599999999999998, 34.4, 43.5, 34.2, 50.2, 57.599999999999994, 78.2, 13.8], [26.400000000000002, 28.599999999999998, 34.4, 43.5, 34.0, 50.0, 58.3, 78.60000000000001, 9.0], [26.200000000000003, 28.4, 34.4, 43.5, 33.4, 49.8, 58.599999999999994, 79.0, 5.0], [26.400000000000002, 28.599999999999998, 34.300000000000004, 43.6, 33.4, 49.7, 58.8, 79.0, 2.3], [26.5, 28.7, 34.699999999999996, 43.3, 33.1, 49.6, 58.599999999999994, 79.3, 1.5], [26.1, 28.499999999999996, 34.300000000000004, 43.3, 33.5, 49.3, 58.599999999999994, 79.60000000000001, 0.4], [25.900000000000002, 28.299999999999997, 34.300000000000004, 43.3, 33.4, 49.2, 58.5, 79.80000000000001, 0.1], [25.8, 28.499999999999996, 34.300000000000004, 43.0, 33.300000000000004, 49.0, 58.5, 80.0, 0.0], [25.8, 28.7, 34.300000000000004, 42.699999999999996, 33.1, 48.8, 58.4, 80.2, 0.0], [25.7, 28.1, 34.0, 42.5, 33.0, 48.9, 58.5, 80.30000000000001, 0.0], [25.6, 27.700000000000003, 34.0, 42.0, 33.0, 48.9, 58.199999999999996, 80.7, 0.0], [25.3, 27.6, 33.300000000000004, 41.699999999999996, 32.300000000000004, 48.9, 58.4, 80.80000000000001, 0.0], [24.9, 27.6, 33.0, 41.6, 32.300000000000004, 49.0, 58.3, 81.39999999999999, 0.0], [24.9, 27.3, 32.800000000000004, 41.4, 32.0, 48.5, 57.99999999999999, 82.1, 0.0], [24.5, 27.1, 33.0, 41.099999999999994, 31.8, 47.8, 57.8, 82.5, 0.0], [24.3, 26.5, 32.800000000000004, 41.0, 31.900000000000002, 47.699999999999996, 57.9, 82.8, 0.0], [24.3, 26.200000000000003, 32.6, 40.699999999999996, 31.4, 47.5, 57.199999999999996, 83.1, 0.0], [24.0, 26.1, 32.2, 40.6, 31.4, 47.8, 57.3, 83.1, 0.0], [23.9, 26.0, 31.7, 40.2, 31.4, 47.099999999999994, 57.3, 83.89999999999999, 0.0], [23.200000000000003, 25.8, 31.7, 39.900000000000006, 30.9, 47.199999999999996, 57.3, 83.6, 0.0], [22.7, 25.8, 31.5, 39.7, 30.599999999999998, 46.6, 56.8, 83.7, 0.0], [22.400000000000002, 25.0, 31.4, 39.1, 30.4, 46.300000000000004, 56.2, 83.7, 0.0], [22.1, 24.4, 30.8, 38.5, 30.0, 46.1, 55.50000000000001, 83.8, 0.0], [21.9, 24.3, 30.4, 38.1, 29.4, 45.4, 55.7, 84.0, 0.0], [21.6, 23.9, 30.0, 37.7, 29.2, 44.7, 55.400000000000006, 83.89999999999999, 0.0], [21.4, 23.599999999999998, 29.599999999999998, 36.9, 28.499999999999996, 44.3, 55.1, 83.89999999999999, 0.0], [20.7, 23.0, 28.999999999999996, 36.3, 28.000000000000004, 44.2, 54.7, 84.1, 0.0], [20.3, 22.7, 28.4, 35.699999999999996, 27.700000000000003, 43.6, 54.300000000000004, 84.39999999999999, 0.0]],
        [[12.5, 12.3, 13.200000000000001, 16.0, 11.0, 15.5, 17.299999999999997, 21.3, 20.599999999999998, 88.3], [13.4, 13.4, 13.3, 17.1, 12.0, 16.900000000000002, 19.2, 23.599999999999998, 22.8, 88.4], [14.099999999999998, 14.2, 14.6, 17.8, 13.0, 18.7, 21.0, 24.9, 24.9, 88.4], [15.2, 14.899999999999999, 15.4, 19.0, 13.8, 19.7, 22.2, 27.0, 27.200000000000003, 88.2], [16.8, 15.6, 16.3, 19.900000000000002, 14.7, 21.4, 24.2, 29.799999999999997, 30.4, 87.7], [17.5, 17.0, 16.900000000000002, 20.9, 15.299999999999999, 23.0, 26.700000000000003, 30.8, 32.7, 87.3], [17.5, 17.5, 17.7, 22.2, 16.400000000000002, 24.5, 28.599999999999998, 32.9, 35.5, 86.9], [18.2, 18.2, 19.3, 23.599999999999998, 17.8, 26.3, 30.3, 35.099999999999994, 38.800000000000004, 86.1], [19.1, 18.9, 20.1, 24.8, 18.7, 27.900000000000002, 31.1, 37.1, 42.1, 85.8], [20.0, 20.4, 21.099999999999998, 25.900000000000002, 19.8, 29.4, 32.2, 39.300000000000004, 44.9, 85.2], [21.2, 20.9, 21.9, 27.0, 20.9, 31.0, 34.0, 41.0, 48.5, 84.7], [21.8, 21.4, 22.3, 28.000000000000004, 21.5, 32.7, 35.9, 42.8, 50.9, 83.2], [22.2, 22.1, 23.400000000000002, 29.4, 23.0, 34.1, 37.5, 44.4, 54.50000000000001, 80.60000000000001], [22.3, 22.6, 24.5, 30.099999999999998, 23.400000000000002, 35.9, 38.3, 47.199999999999996, 57.99999999999999, 77.7], [22.6, 23.0, 25.3, 30.8, 24.6, 36.5, 40.2, 48.5, 60.9, 73.9], [22.8, 24.0, 26.900000000000002, 31.3, 25.2, 37.1, 40.9, 50.4, 63.6, 69.39999999999999], [23.3, 24.5, 27.400000000000002, 32.6, 25.6, 37.8, 42.199999999999996, 51.2, 65.60000000000001, 64.9], [23.7, 25.3, 28.199999999999996, 33.0, 26.1, 38.3, 43.2, 52.400000000000006, 68.30000000000001, 59.0], [24.0, 25.6, 28.499999999999996, 33.800000000000004, 26.900000000000002, 38.9, 43.8, 53.7, 71.1, 50.7], [23.799999999999997, 26.0, 29.4, 34.699999999999996, 28.1, 39.4, 44.0, 54.6, 72.2, 43.2], [24.099999999999998, 25.7, 29.599999999999998, 35.199999999999996, 28.199999999999996, 40.0, 44.7, 54.900000000000006, 73.9, 35.3], [24.0, 25.6, 29.7, 35.5, 28.599999999999998, 40.1, 44.800000000000004, 55.2, 75.4, 28.9], [24.099999999999998, 25.6, 30.3, 35.4, 29.2, 40.5, 44.9, 55.7, 76.1, 22.5], [23.7, 25.5, 30.8, 35.3, 29.4, 40.6, 45.0, 55.60000000000001, 76.6, 17.2], [23.599999999999998, 25.5, 31.0, 35.4, 29.599999999999998, 40.6, 45.0, 55.60000000000001, 77.5, 11.3], [23.5, 25.1, 31.1, 35.699999999999996, 29.2, 40.699999999999996, 44.800000000000004, 55.300000000000004, 77.7, 6.6000000000000005], [23.7, 24.8, 31.3, 35.8, 29.2, 41.099999999999994, 44.9, 55.400000000000006, 78.4, 3.5000000000000004], [23.3, 24.7, 31.1, 35.5, 28.7, 40.9, 44.6, 55.2, 78.60000000000001, 1.4000000000000001], [23.200000000000003, 24.5, 31.0, 35.5, 28.799999999999997, 40.8, 44.3, 55.00000000000001, 78.60000000000001, 0.8999999999999999], [22.900000000000002, 24.3, 30.8, 35.199999999999996, 28.599999999999998, 40.300000000000004, 44.0, 55.2, 78.7, 0.4], [22.7, 24.2, 30.8, 34.9, 28.499999999999996, 40.300000000000004, 43.7, 54.900000000000006, 79.0, 0.1], [22.8, 24.099999999999998, 30.7, 35.0, 28.799999999999997, 39.900000000000006, 43.6, 54.7, 79.5, 0.0], [22.900000000000002, 24.0, 30.2, 34.8, 28.799999999999997, 39.300000000000004, 43.3, 54.900000000000006, 79.7, 0.0], [22.2, 23.599999999999998, 30.0, 34.599999999999994, 28.7, 39.1, 43.1, 54.400000000000006, 80.2, 0.0], [21.8, 23.5, 29.799999999999997, 34.300000000000004, 28.199999999999996, 38.9, 42.8, 54.2, 80.30000000000001, 0.0], [21.6, 23.599999999999998, 29.9, 34.2, 28.299999999999997, 38.9, 42.699999999999996, 53.7, 80.80000000000001, 0.0], [21.5, 23.5, 29.299999999999997, 33.800000000000004, 28.299999999999997, 38.4, 42.4, 53.5, 81.2, 0.0], [21.2, 23.5, 29.4, 33.900000000000006, 27.900000000000002, 37.9, 42.0, 53.1, 81.6, 0.0], [20.9, 23.1, 28.999999999999996, 33.7, 27.800000000000004, 37.6, 41.5, 52.800000000000004, 82.3, 0.0], [20.599999999999998, 23.1, 28.999999999999996, 33.5, 28.000000000000004, 37.3, 40.9, 52.6, 83.2, 0.0], [20.5, 23.0, 28.999999999999996, 33.4, 28.000000000000004, 36.6, 40.1, 52.5, 83.6, 0.0], [20.3, 22.8, 28.7, 33.6, 27.700000000000003, 36.0, 40.2, 52.5, 83.89999999999999, 0.0], [19.8, 22.7, 28.199999999999996, 33.2, 27.6, 36.199999999999996, 39.5, 52.400000000000006, 84.0, 0.0], [19.7, 22.3, 28.1, 33.1, 27.400000000000002, 36.3, 39.2, 51.7, 83.8, 0.0], [19.8, 22.2, 27.800000000000004, 32.9, 27.3, 35.5, 38.9, 51.6, 83.89999999999999, 0.0], [19.400000000000002, 21.6, 27.400000000000002, 32.4, 26.900000000000002, 34.9, 38.3, 51.2, 83.89999999999999, 0.0], [18.8, 21.7, 26.900000000000002, 32.1, 26.700000000000003, 34.4, 37.7, 50.8, 84.3, 0.0], [18.5, 21.4, 26.700000000000003, 31.7, 26.6, 34.1, 37.6, 50.4, 84.8, 0.0], [18.099999999999998, 21.2, 26.400000000000002, 31.4, 26.0, 34.1, 37.1, 50.0, 85.0, 0.0], [17.7, 21.0, 26.200000000000003, 31.0, 25.6, 33.4, 36.7, 49.3, 85.2, 0.0], [17.5, 20.9, 25.900000000000002, 30.2, 25.4, 32.6, 36.0, 48.9, 85.6, 0.0]],
    ]




def main():
    alpha_grid = np.arange(0, 1.001, 0.02)
    accs_clewi = get_accs()
    accs_bic = get_accs_bic()
    accs_derpp = get_accs_derpp()
    accs_er_ace = get_accs_er_ace()

    method_results = {
        'CLeWI+ER': accs_clewi, 
        'CLeWI+BIC': accs_bic, 
        'CLeWI+DER++': accs_derpp, 
        'CLeWI+ER ACE': accs_er_ace
    }

    colors = sns.color_palette("husl", 9)
    with sns.axes_style("darkgrid"):
        for j, method_name in enumerate(method_results):
            plt.subplot(2, 2, j+1)
            accs_method = method_results[method_name]
            for i, accs in enumerate(accs_method):
                accs = np.mean(accs, axis=1)
                plt.plot(alpha_grid, accs, label=f'task {i+1}', color=colors[i])

            plt.legend()
            plt.ylabel('Test accuracy')
            plt.xlabel('Interpolation α')
            plt.title(method_name)
    plt.show()


if __name__ == '__main__':
    main()
