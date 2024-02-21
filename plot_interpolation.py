import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_accs():
    "clewi_er_ace alpha 0.2"
    return [
        [[73.7, 85.9], [76.0, 84.1], [77.60000000000001, 83.0], [79.0, 80.9], [81.10000000000001, 79.5], [81.89999999999999, 76.8], [83.0, 74.3], [84.2, 70.1], [85.0, 66.10000000000001], [85.9, 61.7], [86.4, 56.99999999999999], [86.8, 52.2], [87.4, 45.800000000000004], [87.6, 39.2], [87.6, 33.800000000000004], [87.7, 28.4], [87.6, 22.2], [87.9, 16.2], [87.7, 12.1], [87.8, 8.5], [87.5, 5.1], [87.7, 2.3], [87.6, 1.0999999999999999], [87.8, 0.5], [88.0, 0.1], [87.9, 0.0], [88.1, 0.0], [88.1, 0.0], [87.9, 0.0], [87.8, 0.0], [87.6, 0.0], [87.7, 0.0], [87.6, 0.0], [87.9, 0.0], [87.8, 0.0], [87.8, 0.0], [88.3, 0.0], [88.3, 0.0], [88.0, 0.0], [88.0, 0.0], [87.9, 0.0], [88.1, 0.0], [88.2, 0.0], [88.0, 0.0], [88.1, 0.0], [88.2, 0.0], [88.2, 0.0], [88.2, 0.0], [88.1, 0.0], [87.9, 0.0], [87.7, 0.0]],
        [[64.2, 65.0, 76.6], [66.10000000000001, 66.7, 74.0], [68.2, 69.39999999999999, 71.2], [69.6, 71.5, 68.5], [70.8, 73.2, 65.8], [72.7, 73.6, 61.9], [73.9, 74.5, 57.8], [75.4, 75.2, 54.6], [76.7, 76.0, 50.6], [77.7, 77.5, 47.099999999999994], [78.3, 77.8, 41.6], [78.7, 78.2, 37.3], [79.2, 78.4, 32.300000000000004], [79.9, 78.0, 26.5], [80.30000000000001, 77.9, 21.3], [80.5, 77.7, 16.7], [80.80000000000001, 77.5, 12.2], [81.10000000000001, 77.3, 8.3], [81.39999999999999, 77.10000000000001, 5.6000000000000005], [81.6, 76.9, 3.5999999999999996], [81.69999999999999, 77.4, 2.4], [82.0, 77.0, 1.0999999999999999], [82.39999999999999, 77.2, 0.5], [82.5, 76.7, 0.2], [82.5, 76.3, 0.2], [82.5, 75.8, 0.2], [82.69999999999999, 75.6, 0.1], [82.89999999999999, 75.2, 0.0], [82.8, 74.7, 0.0], [83.1, 74.1, 0.0], [83.0, 73.3, 0.0], [83.39999999999999, 72.5, 0.0], [83.7, 71.7, 0.0], [83.89999999999999, 71.39999999999999, 0.0], [84.2, 70.8, 0.0], [84.2, 69.89999999999999, 0.0], [84.6, 69.5, 0.0], [84.6, 69.0, 0.0], [84.89999999999999, 68.0, 0.0], [85.2, 67.5, 0.0], [85.3, 66.7, 0.0], [85.5, 65.60000000000001, 0.0], [85.3, 64.7, 0.0], [85.39999999999999, 63.800000000000004, 0.0], [85.3, 62.3, 0.0], [85.1, 60.8, 0.0], [85.3, 59.5, 0.0], [84.6, 57.99999999999999, 0.0], [84.5, 56.8, 0.0], [84.5, 54.400000000000006, 0.0], [84.3, 53.5, 0.0]],
        [[55.50000000000001, 58.4, 58.699999999999996, 76.7], [57.3, 60.199999999999996, 59.9, 73.8], [58.9, 61.7, 60.6, 71.6], [60.699999999999996, 62.9, 61.0, 68.5], [62.2, 64.3, 61.4, 64.7], [63.4, 65.3, 61.8, 60.4], [64.2, 66.4, 62.0, 55.7], [65.10000000000001, 67.30000000000001, 62.1, 49.4], [66.10000000000001, 68.30000000000001, 62.6, 44.7], [66.7, 69.3, 62.4, 38.6], [67.5, 69.69999999999999, 62.7, 32.4], [68.4, 70.0, 63.0, 27.0], [68.8, 70.3, 63.0, 22.1], [69.3, 70.3, 62.7, 18.8], [69.39999999999999, 70.8, 62.5, 13.900000000000002], [70.0, 70.89999999999999, 62.5, 11.4], [70.19999999999999, 71.5, 61.7, 7.3], [70.19999999999999, 72.1, 61.6, 4.8], [70.8, 72.3, 61.3, 3.3000000000000003], [71.39999999999999, 72.89999999999999, 61.3, 1.7000000000000002], [71.39999999999999, 72.89999999999999, 60.699999999999996, 0.5], [71.8, 73.0, 61.0, 0.0], [71.89999999999999, 73.3, 60.0, 0.0], [72.0, 73.8, 59.199999999999996, 0.0], [72.7, 74.0, 58.5, 0.0], [72.7, 74.0, 58.099999999999994, 0.0], [72.8, 74.6, 57.699999999999996, 0.0], [73.0, 74.9, 57.199999999999996, 0.0], [73.5, 75.1, 56.599999999999994, 0.0], [73.6, 75.6, 55.900000000000006, 0.0], [73.8, 75.8, 54.800000000000004, 0.0], [74.1, 76.1, 54.300000000000004, 0.0], [74.0, 76.6, 53.400000000000006, 0.0], [74.1, 76.7, 52.7, 0.0], [74.2, 76.9, 52.0, 0.0], [74.5, 77.0, 51.1, 0.0], [74.7, 76.6, 50.0, 0.0], [74.7, 76.7, 49.4, 0.0], [74.6, 76.7, 48.199999999999996, 0.0], [75.3, 77.10000000000001, 47.199999999999996, 0.0], [75.5, 77.3, 46.5, 0.0], [75.3, 77.8, 45.6, 0.0], [75.1, 78.0, 44.5, 0.0], [75.0, 77.60000000000001, 43.8, 0.0], [75.1, 77.5, 43.2, 0.0], [75.0, 77.3, 42.4, 0.0], [74.7, 77.2, 41.699999999999996, 0.0], [74.8, 77.10000000000001, 41.099999999999994, 0.0], [74.2, 76.8, 39.900000000000006, 0.0], [74.3, 76.8, 39.2, 0.0], [74.2, 76.3, 37.9, 0.0]],
        [[46.7, 47.5, 48.6, 57.4, 69.69999999999999], [48.0, 49.2, 49.7, 58.4, 66.2], [49.7, 50.9, 50.8, 59.0, 63.5], [50.9, 52.800000000000004, 52.2, 59.5, 60.5], [52.2, 54.0, 53.5, 59.5, 55.900000000000006], [53.1, 56.3, 54.50000000000001, 60.199999999999996, 52.300000000000004], [54.0, 57.199999999999996, 55.50000000000001, 60.699999999999996, 47.8], [55.300000000000004, 58.9, 56.3, 61.199999999999996, 43.1], [56.3, 60.099999999999994, 56.599999999999994, 61.4, 38.800000000000004], [57.199999999999996, 60.699999999999996, 57.49999999999999, 61.199999999999996, 33.5], [57.9, 61.7, 58.199999999999996, 61.3, 29.4], [58.5, 62.3, 58.9, 61.5, 24.4], [59.699999999999996, 62.7, 59.0, 60.9, 19.2], [60.099999999999994, 63.3, 59.199999999999996, 60.5, 15.1], [60.699999999999996, 63.5, 59.599999999999994, 60.099999999999994, 11.600000000000001], [61.4, 64.3, 60.099999999999994, 59.699999999999996, 7.3], [61.6, 65.0, 60.099999999999994, 59.3, 6.1], [61.7, 65.3, 61.0, 58.9, 3.9], [61.7, 65.4, 61.6, 58.699999999999996, 2.1999999999999997], [61.6, 65.7, 62.2, 58.8, 1.2], [62.1, 65.7, 62.2, 58.5, 0.4], [62.6, 66.10000000000001, 62.5, 57.9, 0.2], [62.9, 66.2, 62.4, 57.3, 0.0], [63.3, 67.0, 62.5, 56.39999999999999, 0.0], [63.6, 66.8, 62.8, 55.800000000000004, 0.0], [63.9, 67.10000000000001, 63.2, 55.300000000000004, 0.0], [63.9, 67.5, 63.3, 54.800000000000004, 0.0], [64.1, 67.80000000000001, 63.5, 54.0, 0.0], [64.3, 68.0, 63.6, 53.400000000000006, 0.0], [64.3, 68.2, 63.6, 52.5, 0.0], [64.5, 68.2, 63.7, 51.9, 0.0], [64.8, 68.2, 64.0, 50.7, 0.0], [64.9, 68.0, 63.800000000000004, 50.0, 0.0], [65.0, 67.9, 64.2, 49.5, 0.0], [65.3, 68.10000000000001, 63.9, 48.699999999999996, 0.0], [65.7, 68.5, 63.800000000000004, 47.9, 0.0], [65.60000000000001, 68.5, 63.800000000000004, 47.5, 0.0], [65.5, 68.5, 63.800000000000004, 46.9, 0.0], [65.2, 68.7, 63.800000000000004, 46.0, 0.0], [65.60000000000001, 68.4, 63.9, 45.2, 0.0], [66.0, 68.30000000000001, 63.6, 44.4, 0.0], [66.10000000000001, 68.30000000000001, 63.9, 43.4, 0.0], [66.0, 68.10000000000001, 63.1, 42.6, 0.0], [66.7, 67.7, 63.4, 41.6, 0.0], [66.8, 67.9, 63.2, 39.4, 0.0], [66.8, 67.7, 62.7, 37.9, 0.0], [66.8, 67.4, 62.5, 36.0, 0.0], [66.5, 67.2, 62.3, 34.699999999999996, 0.0], [66.3, 66.60000000000001, 62.0, 33.900000000000006, 0.0], [66.3, 66.2, 61.8, 32.7, 0.0], [65.7, 66.10000000000001, 61.3, 31.2, 0.0]],
        [[44.5, 47.199999999999996, 40.8, 50.6, 46.1, 74.5], [46.2, 48.199999999999996, 41.9, 51.800000000000004, 48.3, 71.6], [47.5, 49.9, 42.8, 52.800000000000004, 50.0, 67.80000000000001], [48.3, 50.8, 43.7, 54.0, 50.5, 64.1], [49.5, 51.800000000000004, 44.6, 54.2, 51.0, 60.099999999999994], [50.4, 52.5, 45.800000000000004, 54.800000000000004, 51.9, 55.900000000000006], [50.7, 53.6, 46.7, 56.10000000000001, 52.300000000000004, 50.6], [51.300000000000004, 54.50000000000001, 47.5, 56.599999999999994, 52.400000000000006, 44.9], [51.800000000000004, 54.6, 47.9, 57.599999999999994, 52.400000000000006, 39.300000000000004], [52.6, 55.00000000000001, 48.3, 58.099999999999994, 52.5, 34.699999999999996], [53.2, 55.7, 48.6, 58.599999999999994, 52.2, 28.499999999999996], [53.900000000000006, 56.10000000000001, 48.8, 58.9, 51.800000000000004, 22.900000000000002], [54.1, 56.8, 48.9, 59.4, 51.0, 16.5], [54.6, 56.89999999999999, 49.6, 59.599999999999994, 51.0, 12.5], [54.7, 57.199999999999996, 49.8, 59.5, 51.0, 8.799999999999999], [55.2, 57.599999999999994, 49.9, 59.9, 50.5, 4.9], [55.1, 57.699999999999996, 50.2, 59.699999999999996, 49.7, 3.2], [55.50000000000001, 57.99999999999999, 50.6, 59.5, 49.1, 1.7999999999999998], [56.00000000000001, 58.099999999999994, 50.8, 59.599999999999994, 48.8, 0.4], [56.39999999999999, 58.4, 51.0, 59.599999999999994, 48.199999999999996, 0.3], [56.2, 58.8, 51.6, 59.699999999999996, 47.4, 0.1], [56.49999999999999, 58.9, 52.300000000000004, 59.8, 46.6, 0.0], [56.99999999999999, 59.199999999999996, 52.1, 59.8, 46.6, 0.0], [57.199999999999996, 59.3, 52.800000000000004, 59.9, 46.0, 0.0], [57.49999999999999, 59.5, 53.400000000000006, 59.9, 45.300000000000004, 0.0], [57.4, 59.599999999999994, 53.5, 60.0, 44.5, 0.0], [57.599999999999994, 59.599999999999994, 53.900000000000006, 60.5, 44.0, 0.0], [57.49999999999999, 59.5, 54.50000000000001, 60.6, 43.7, 0.0], [57.49999999999999, 59.699999999999996, 54.800000000000004, 60.6, 43.2, 0.0], [57.49999999999999, 59.699999999999996, 54.900000000000006, 60.8, 42.199999999999996, 0.0], [57.49999999999999, 60.4, 55.2, 60.8, 41.699999999999996, 0.0], [57.9, 60.199999999999996, 55.7, 60.6, 41.6, 0.0], [57.9, 60.699999999999996, 55.900000000000006, 60.5, 41.099999999999994, 0.0], [57.8, 60.6, 55.800000000000004, 60.9, 40.6, 0.0], [57.9, 60.4, 55.900000000000006, 60.6, 40.300000000000004, 0.0], [57.9, 60.199999999999996, 55.800000000000004, 60.4, 39.900000000000006, 0.0], [58.099999999999994, 59.9, 55.7, 60.6, 38.9, 0.0], [57.9, 60.0, 55.60000000000001, 60.4, 38.2, 0.0], [57.699999999999996, 60.099999999999994, 55.50000000000001, 60.3, 37.5, 0.0], [57.49999999999999, 60.5, 55.400000000000006, 60.199999999999996, 36.7, 0.0], [57.3, 60.9, 55.50000000000001, 60.099999999999994, 35.8, 0.0], [57.4, 61.0, 55.900000000000006, 60.099999999999994, 35.3, 0.0], [57.49999999999999, 61.199999999999996, 56.2, 59.5, 34.699999999999996, 0.0], [57.3, 61.0, 56.39999999999999, 59.4, 33.800000000000004, 0.0], [57.099999999999994, 61.199999999999996, 56.49999999999999, 59.199999999999996, 33.1, 0.0], [56.2, 61.0, 56.49999999999999, 59.5, 32.1, 0.0], [56.00000000000001, 60.8, 56.2, 59.099999999999994, 31.5, 0.0], [55.7, 60.699999999999996, 56.2, 58.8, 30.7, 0.0], [54.800000000000004, 60.6, 56.00000000000001, 58.3, 30.0, 0.0], [54.50000000000001, 60.3, 55.800000000000004, 57.8, 29.2, 0.0], [54.2, 60.5, 55.800000000000004, 57.599999999999994, 28.499999999999996, 0.0]],
        [[41.6, 41.5, 39.5, 43.3, 40.300000000000004, 54.400000000000006, 73.7], [42.8, 42.8, 40.699999999999996, 45.6, 42.3, 55.2, 70.6], [43.2, 43.9, 41.8, 46.800000000000004, 43.8, 55.7, 66.60000000000001], [44.5, 45.4, 42.6, 47.9, 44.800000000000004, 55.50000000000001, 62.3], [45.800000000000004, 46.800000000000004, 43.8, 49.7, 45.4, 55.800000000000004, 58.3], [46.5, 47.599999999999994, 44.6, 50.8, 45.6, 56.3, 53.7], [46.7, 48.5, 45.2, 51.4, 46.400000000000006, 56.49999999999999, 48.4], [47.0, 49.2, 45.6, 52.5, 46.9, 56.599999999999994, 43.4], [47.699999999999996, 49.8, 45.800000000000004, 52.7, 46.7, 56.599999999999994, 36.5], [47.8, 49.9, 46.400000000000006, 53.1, 46.9, 56.599999999999994, 31.0], [48.0, 50.6, 46.800000000000004, 53.5, 47.199999999999996, 56.699999999999996, 25.6], [48.199999999999996, 51.2, 46.800000000000004, 53.800000000000004, 47.4, 56.89999999999999, 19.1], [48.3, 51.7, 47.099999999999994, 54.2, 47.599999999999994, 56.8, 14.7], [48.699999999999996, 51.800000000000004, 47.4, 54.6, 48.1, 56.599999999999994, 10.5], [49.2, 52.300000000000004, 47.699999999999996, 55.1, 48.3, 55.900000000000006, 6.800000000000001], [49.7, 52.7, 48.0, 55.2, 48.1, 55.2, 4.2], [50.0, 53.0, 48.0, 55.7, 48.0, 54.7, 1.7999999999999998], [50.8, 53.1, 48.1, 55.7, 48.199999999999996, 54.7, 0.7000000000000001], [51.4, 53.300000000000004, 47.8, 55.60000000000001, 48.699999999999996, 54.6, 0.6], [51.4, 53.300000000000004, 48.0, 55.900000000000006, 48.5, 54.1, 0.2], [52.2, 53.7, 47.9, 56.99999999999999, 49.0, 53.7, 0.0], [52.1, 53.800000000000004, 48.0, 57.4, 49.2, 53.2, 0.0], [52.400000000000006, 54.1, 48.199999999999996, 57.599999999999994, 49.6, 52.800000000000004, 0.0], [52.1, 54.2, 49.1, 57.599999999999994, 49.7, 52.0, 0.0], [52.400000000000006, 54.6, 49.3, 57.49999999999999, 50.0, 51.4, 0.0], [52.6, 54.300000000000004, 49.3, 57.3, 49.9, 51.2, 0.0], [52.7, 54.2, 49.7, 57.49999999999999, 50.2, 50.6, 0.0], [52.400000000000006, 54.1, 49.7, 56.99999999999999, 50.4, 50.4, 0.0], [52.6, 54.400000000000006, 49.6, 57.099999999999994, 50.5, 49.7, 0.0], [52.800000000000004, 54.7, 49.3, 57.4, 50.8, 49.1, 0.0], [52.900000000000006, 54.400000000000006, 49.5, 57.3, 50.9, 48.199999999999996, 0.0], [53.5, 54.50000000000001, 49.7, 56.99999999999999, 51.4, 47.4, 0.0], [53.5, 54.50000000000001, 50.0, 56.699999999999996, 51.9, 46.6, 0.0], [53.400000000000006, 54.6, 50.1, 56.3, 52.2, 45.7, 0.0], [53.400000000000006, 54.7, 50.4, 56.3, 52.2, 45.2, 0.0], [53.7, 54.800000000000004, 50.7, 56.10000000000001, 52.1, 44.800000000000004, 0.0], [53.300000000000004, 54.800000000000004, 50.7, 56.00000000000001, 51.6, 44.5, 0.0], [53.300000000000004, 55.2, 50.7, 56.00000000000001, 51.2, 43.2, 0.0], [53.400000000000006, 55.00000000000001, 50.8, 55.7, 51.0, 42.199999999999996, 0.0], [53.800000000000004, 55.1, 50.3, 56.00000000000001, 51.0, 41.3, 0.0], [53.7, 55.00000000000001, 50.1, 55.900000000000006, 50.8, 40.6, 0.0], [53.5, 54.900000000000006, 49.8, 55.7, 50.9, 38.7, 0.0], [53.0, 55.2, 49.8, 55.300000000000004, 50.9, 37.8, 0.0], [52.900000000000006, 55.300000000000004, 49.6, 55.60000000000001, 51.0, 37.2, 0.0], [52.800000000000004, 55.400000000000006, 49.5, 55.300000000000004, 50.7, 35.9, 0.0], [52.7, 54.900000000000006, 49.1, 54.50000000000001, 50.6, 34.599999999999994, 0.0], [52.6, 55.300000000000004, 49.2, 54.300000000000004, 50.6, 33.0, 0.0], [52.5, 55.50000000000001, 49.0, 54.2, 50.3, 32.0, 0.0], [52.800000000000004, 55.60000000000001, 49.3, 54.0, 50.1, 30.9, 0.0], [52.5, 55.400000000000006, 49.0, 53.6, 49.9, 29.099999999999998, 0.0], [52.6, 55.1, 49.1, 53.7, 49.2, 28.299999999999997, 0.0]],
        [[39.4, 38.3, 29.599999999999998, 43.4, 36.6, 47.199999999999996, 50.9, 73.2], [40.300000000000004, 39.7, 31.2, 45.1, 38.1, 49.3, 51.6, 69.19999999999999], [42.4, 41.199999999999996, 32.5, 45.800000000000004, 39.0, 50.8, 52.1, 65.9], [44.1, 42.3, 33.2, 46.400000000000006, 40.2, 52.2, 52.7, 61.7], [45.2, 43.0, 34.0, 47.699999999999996, 41.0, 53.2, 53.2, 57.099999999999994], [46.1, 43.7, 34.8, 48.4, 42.9, 54.0, 53.6, 50.9], [47.3, 44.3, 35.9, 49.4, 43.8, 54.800000000000004, 53.5, 45.4], [47.4, 44.5, 36.5, 49.8, 44.4, 55.1, 53.400000000000006, 40.400000000000006], [47.9, 44.800000000000004, 37.2, 49.9, 45.300000000000004, 55.900000000000006, 53.6, 34.1], [48.4, 45.6, 37.4, 50.4, 45.6, 56.49999999999999, 53.400000000000006, 27.500000000000004], [48.8, 46.0, 38.2, 50.7, 46.0, 57.3, 53.400000000000006, 21.6], [49.0, 47.0, 39.0, 50.8, 46.9, 57.8, 53.400000000000006, 16.3], [49.5, 47.5, 39.4, 51.2, 47.699999999999996, 58.099999999999994, 53.1, 12.3], [49.8, 47.699999999999996, 40.300000000000004, 51.9, 47.8, 58.4, 52.900000000000006, 10.2], [49.8, 48.0, 40.9, 52.5, 48.4, 58.9, 52.800000000000004, 6.1], [50.0, 48.199999999999996, 41.4, 53.0, 47.9, 59.099999999999994, 52.6, 3.3000000000000003], [50.5, 48.5, 41.6, 53.2, 48.0, 59.099999999999994, 52.2, 1.3], [50.3, 48.8, 41.8, 53.400000000000006, 48.1, 59.5, 51.9, 0.7000000000000001], [50.5, 48.8, 42.1, 53.6, 48.0, 59.9, 52.0, 0.4], [50.4, 49.2, 42.699999999999996, 53.6, 47.4, 60.4, 51.300000000000004, 0.2], [50.5, 49.1, 43.1, 53.800000000000004, 47.599999999999994, 60.3, 50.9, 0.1], [50.7, 49.2, 43.3, 53.5, 48.0, 60.199999999999996, 50.0, 0.0], [51.2, 49.2, 43.8, 53.5, 48.199999999999996, 61.0, 49.6, 0.0], [51.4, 49.7, 44.1, 53.6, 48.1, 60.9, 48.5, 0.0], [51.5, 49.5, 44.7, 53.7, 48.199999999999996, 60.8, 47.9, 0.0], [51.4, 49.4, 44.5, 53.800000000000004, 48.4, 60.699999999999996, 47.4, 0.0], [51.5, 49.6, 44.3, 54.0, 48.4, 60.6, 46.7, 0.0], [51.4, 49.9, 45.300000000000004, 54.0, 48.5, 60.9, 46.0, 0.0], [51.6, 49.7, 45.300000000000004, 54.1, 48.5, 60.9, 45.7, 0.0], [51.6, 49.7, 45.4, 54.2, 48.6, 61.1, 45.5, 0.0], [51.2, 50.2, 45.9, 54.1, 48.4, 61.0, 44.0, 0.0], [51.300000000000004, 50.3, 45.9, 54.2, 48.5, 61.0, 43.6, 0.0], [51.2, 50.5, 46.300000000000004, 54.6, 48.1, 60.8, 42.9, 0.0], [51.2, 50.4, 46.7, 54.300000000000004, 48.3, 60.199999999999996, 41.3, 0.0], [50.9, 51.0, 47.199999999999996, 54.300000000000004, 48.1, 60.0, 40.5, 0.0], [50.3, 51.300000000000004, 47.199999999999996, 54.2, 47.699999999999996, 59.599999999999994, 39.800000000000004, 0.0], [50.0, 51.5, 46.7, 54.300000000000004, 47.699999999999996, 59.599999999999994, 38.1, 0.0], [50.0, 51.6, 46.7, 54.0, 47.5, 59.5, 36.9, 0.0], [50.0, 51.9, 46.800000000000004, 54.2, 47.4, 59.4, 36.3, 0.0], [49.6, 51.6, 46.5, 53.800000000000004, 47.0, 59.0, 35.6, 0.0], [49.5, 52.1, 46.1, 53.800000000000004, 46.9, 58.599999999999994, 34.1, 0.0], [49.2, 52.0, 46.0, 53.400000000000006, 46.9, 58.5, 32.9, 0.0], [49.3, 52.0, 46.2, 53.400000000000006, 46.800000000000004, 57.99999999999999, 31.6, 0.0], [49.2, 52.0, 45.9, 53.6, 46.6, 57.699999999999996, 30.599999999999998, 0.0], [48.9, 52.2, 45.9, 53.300000000000004, 46.7, 57.49999999999999, 28.599999999999998, 0.0], [48.9, 52.2, 45.6, 53.2, 46.300000000000004, 57.199999999999996, 27.6, 0.0], [48.6, 51.9, 45.300000000000004, 52.900000000000006, 46.1, 56.8, 26.6, 0.0], [48.5, 51.800000000000004, 45.300000000000004, 52.7, 45.6, 56.49999999999999, 25.7, 0.0], [48.4, 51.7, 45.6, 52.6, 45.300000000000004, 56.3, 24.7, 0.0], [47.8, 51.7, 45.300000000000004, 52.5, 44.6, 55.900000000000006, 23.400000000000002, 0.0], [47.599999999999994, 51.5, 45.2, 52.1, 44.3, 55.50000000000001, 22.3, 0.0]],
        [[36.4, 33.800000000000004, 29.799999999999997, 36.0, 34.699999999999996, 36.9, 41.6, 49.8, 68.10000000000001], [38.1, 34.599999999999994, 30.9, 37.0, 35.5, 38.800000000000004, 43.8, 50.8, 63.5], [39.1, 35.8, 32.4, 38.2, 36.1, 40.0, 45.2, 51.6, 60.3], [39.900000000000006, 36.5, 33.2, 39.6, 37.0, 41.5, 46.300000000000004, 52.7, 56.3], [41.4, 37.7, 34.1, 40.6, 37.1, 42.8, 46.9, 52.800000000000004, 51.7], [42.5, 38.0, 34.599999999999994, 41.9, 37.7, 43.9, 47.699999999999996, 53.400000000000006, 46.400000000000006], [42.8, 38.7, 34.9, 42.699999999999996, 38.1, 45.0, 48.5, 54.2, 39.800000000000004], [43.7, 39.1, 34.9, 43.8, 38.2, 45.9, 49.7, 54.2, 33.5], [44.5, 39.4, 35.0, 44.800000000000004, 38.4, 46.9, 50.7, 53.7, 26.8], [44.9, 39.800000000000004, 34.9, 45.9, 38.4, 47.3, 51.2, 53.400000000000006, 21.4], [45.2, 40.1, 35.0, 46.300000000000004, 38.800000000000004, 48.0, 51.800000000000004, 52.400000000000006, 16.0], [45.4, 40.1, 35.099999999999994, 46.400000000000006, 39.4, 48.5, 52.0, 52.6, 10.8], [45.800000000000004, 40.2, 35.6, 47.099999999999994, 39.6, 49.3, 52.2, 52.2, 7.1], [46.2, 40.6, 36.1, 46.9, 40.5, 49.3, 52.400000000000006, 51.6, 4.3999999999999995], [46.2, 40.8, 36.4, 46.9, 40.8, 49.5, 52.6, 51.1, 2.6], [46.5, 41.699999999999996, 36.7, 47.4, 41.0, 49.6, 52.800000000000004, 50.4, 1.3], [46.400000000000006, 42.0, 36.9, 47.8, 41.3, 50.0, 53.400000000000006, 50.1, 0.4], [46.6, 42.5, 36.9, 47.9, 41.3, 50.3, 53.800000000000004, 49.6, 0.1], [46.800000000000004, 42.699999999999996, 37.3, 47.9, 40.9, 50.3, 54.0, 49.0, 0.1], [47.3, 42.9, 37.8, 48.199999999999996, 40.9, 50.8, 54.400000000000006, 48.3, 0.0], [47.3, 43.2, 37.6, 48.3, 40.9, 51.1, 54.2, 47.4, 0.0], [47.599999999999994, 43.2, 37.5, 48.9, 41.0, 51.300000000000004, 54.400000000000006, 46.7, 0.0], [47.4, 43.6, 37.4, 49.3, 41.5, 51.4, 54.7, 46.0, 0.0], [47.599999999999994, 44.0, 37.6, 49.5, 41.699999999999996, 51.800000000000004, 55.00000000000001, 45.4, 0.0], [47.3, 44.1, 38.1, 49.8, 41.699999999999996, 52.1, 55.400000000000006, 45.1, 0.0], [47.699999999999996, 44.5, 37.6, 49.7, 41.5, 52.300000000000004, 55.300000000000004, 44.2, 0.0], [48.1, 44.5, 37.3, 49.9, 41.8, 52.5, 55.2, 43.5, 0.0], [48.3, 44.5, 37.5, 50.4, 42.0, 52.800000000000004, 55.50000000000001, 42.699999999999996, 0.0], [48.199999999999996, 45.2, 37.5, 50.6, 42.1, 52.900000000000006, 55.50000000000001, 41.9, 0.0], [48.5, 45.4, 37.6, 50.5, 42.199999999999996, 53.300000000000004, 55.60000000000001, 41.4, 0.0], [48.4, 45.300000000000004, 37.5, 50.8, 42.3, 53.800000000000004, 55.300000000000004, 40.699999999999996, 0.0], [48.3, 45.800000000000004, 37.2, 51.1, 42.5, 54.0, 55.2, 40.1, 0.0], [48.1, 45.9, 37.4, 51.5, 42.699999999999996, 54.400000000000006, 55.00000000000001, 39.0, 0.0], [48.4, 46.1, 37.4, 51.4, 42.8, 54.6, 54.900000000000006, 38.6, 0.0], [48.8, 46.2, 37.5, 51.0, 42.699999999999996, 54.800000000000004, 54.800000000000004, 38.0, 0.0], [49.0, 46.0, 37.5, 50.8, 42.9, 55.300000000000004, 54.400000000000006, 37.6, 0.0], [49.2, 46.2, 37.5, 50.9, 43.0, 55.300000000000004, 55.00000000000001, 37.1, 0.0], [49.1, 46.5, 37.3, 51.0, 43.3, 55.400000000000006, 54.50000000000001, 36.199999999999996, 0.0], [48.5, 46.2, 37.7, 50.8, 43.3, 55.2, 54.2, 35.0, 0.0], [48.3, 46.6, 37.4, 50.4, 43.5, 55.1, 54.2, 34.2, 0.0], [48.199999999999996, 46.5, 37.4, 49.8, 43.3, 55.300000000000004, 54.0, 33.6, 0.0], [47.8, 46.7, 37.4, 49.8, 43.4, 55.7, 54.1, 33.1, 0.0], [47.599999999999994, 46.6, 37.4, 49.3, 43.4, 56.00000000000001, 53.7, 32.4, 0.0], [47.699999999999996, 46.800000000000004, 37.1, 49.2, 43.5, 56.00000000000001, 53.6, 30.8, 0.0], [47.699999999999996, 47.099999999999994, 37.4, 49.2, 43.5, 56.10000000000001, 53.5, 29.599999999999998, 0.0], [47.8, 47.3, 37.4, 48.6, 43.6, 56.3, 53.2, 28.799999999999997, 0.0], [47.9, 47.099999999999994, 37.4, 48.1, 43.4, 56.599999999999994, 53.0, 27.500000000000004, 0.0], [47.8, 46.7, 37.2, 48.1, 42.8, 56.3, 52.800000000000004, 26.1, 0.0], [47.8, 46.7, 37.1, 48.0, 43.0, 56.49999999999999, 52.7, 24.8, 0.0], [47.699999999999996, 46.6, 36.8, 47.5, 42.9, 55.900000000000006, 52.300000000000004, 23.5, 0.0], [47.199999999999996, 46.300000000000004, 36.3, 47.099999999999994, 42.699999999999996, 55.7, 51.6, 23.200000000000003, 0.0]],
        [[34.9, 32.0, 26.400000000000002, 35.4, 25.7, 39.0, 45.300000000000004, 43.4, 47.099999999999994, 59.0], [35.199999999999996, 32.6, 27.6, 36.1, 26.3, 39.7, 45.800000000000004, 44.1, 47.699999999999996, 54.800000000000004], [35.9, 33.6, 28.7, 36.0, 27.1, 40.0, 46.300000000000004, 45.1, 47.8, 49.6], [36.6, 34.300000000000004, 29.099999999999998, 36.9, 28.000000000000004, 40.699999999999996, 46.7, 46.400000000000006, 48.199999999999996, 45.1], [36.9, 34.4, 30.2, 37.4, 28.799999999999997, 42.0, 47.4, 47.4, 48.1, 39.6], [37.1, 34.599999999999994, 31.0, 38.1, 29.7, 42.699999999999996, 48.3, 48.1, 47.699999999999996, 34.4], [38.2, 34.9, 31.3, 38.4, 30.3, 43.4, 48.699999999999996, 49.0, 47.0, 29.799999999999997], [38.2, 35.8, 31.900000000000002, 38.5, 31.1, 44.1, 48.9, 49.6, 46.6, 25.3], [38.6, 36.199999999999996, 32.6, 38.7, 31.900000000000002, 45.0, 49.2, 50.1, 46.400000000000006, 19.8], [38.7, 36.3, 33.300000000000004, 39.0, 32.5, 45.1, 49.5, 50.6, 45.9, 15.5], [39.300000000000004, 36.6, 33.7, 39.1, 33.6, 45.6, 49.5, 51.300000000000004, 45.6, 11.0], [39.6, 36.8, 33.800000000000004, 39.6, 34.2, 46.2, 50.1, 51.9, 45.4, 7.6], [40.0, 36.7, 33.7, 39.900000000000006, 34.300000000000004, 46.5, 50.3, 52.0, 44.9, 5.5], [40.6, 37.3, 34.5, 39.800000000000004, 34.2, 46.800000000000004, 50.5, 52.1, 45.0, 3.9], [40.9, 37.5, 35.0, 39.900000000000006, 34.300000000000004, 47.0, 51.300000000000004, 52.6, 44.800000000000004, 1.7000000000000002], [41.199999999999996, 37.8, 35.3, 40.0, 34.2, 47.099999999999994, 51.6, 53.0, 44.1, 0.8999999999999999], [41.099999999999994, 38.0, 35.699999999999996, 40.699999999999996, 34.1, 47.4, 52.2, 52.6, 43.2, 0.3], [41.4, 38.1, 35.9, 40.699999999999996, 34.2, 47.9, 52.0, 52.5, 42.8, 0.0], [41.8, 38.5, 36.1, 40.8, 34.5, 47.9, 52.2, 52.5, 42.3, 0.0], [42.0, 39.1, 36.3, 41.199999999999996, 34.599999999999994, 48.6, 52.7, 52.900000000000006, 41.699999999999996, 0.0], [42.3, 38.9, 36.4, 41.5, 34.5, 48.699999999999996, 53.1, 53.300000000000004, 41.3, 0.0], [42.5, 39.1, 36.3, 41.699999999999996, 34.8, 48.699999999999996, 52.400000000000006, 53.400000000000006, 40.5, 0.0], [42.3, 39.300000000000004, 36.3, 42.3, 35.099999999999994, 48.6, 52.5, 53.1, 40.1, 0.0], [42.8, 39.800000000000004, 36.5, 42.5, 35.099999999999994, 48.9, 52.5, 53.1, 39.800000000000004, 0.0], [43.1, 39.900000000000006, 36.5, 42.8, 35.4, 49.2, 52.7, 53.6, 38.9, 0.0], [43.2, 40.0, 36.3, 42.8, 35.4, 49.1, 52.6, 53.5, 38.0, 0.0], [43.5, 39.900000000000006, 36.1, 42.6, 35.699999999999996, 49.3, 52.6, 53.900000000000006, 37.3, 0.0], [44.0, 40.2, 35.8, 42.4, 36.1, 49.7, 52.6, 53.800000000000004, 36.3, 0.0], [43.9, 40.2, 35.5, 42.6, 36.8, 49.8, 52.400000000000006, 54.1, 36.0, 0.0], [43.9, 40.300000000000004, 35.6, 42.8, 37.2, 50.0, 52.7, 54.1, 35.5, 0.0], [44.1, 39.800000000000004, 35.099999999999994, 42.5, 37.1, 50.0, 52.800000000000004, 54.2, 34.599999999999994, 0.0], [44.3, 39.7, 35.0, 43.1, 37.4, 50.1, 53.0, 54.7, 33.900000000000006, 0.0], [44.1, 40.0, 35.0, 43.4, 37.5, 50.0, 52.800000000000004, 54.7, 33.300000000000004, 0.0], [43.6, 40.1, 34.9, 43.3, 37.5, 50.0, 52.900000000000006, 54.6, 32.2, 0.0], [43.6, 40.400000000000006, 35.199999999999996, 43.4, 37.0, 50.3, 53.1, 54.2, 30.9, 0.0], [43.8, 40.300000000000004, 35.199999999999996, 43.9, 37.2, 50.5, 53.0, 54.1, 30.0, 0.0], [44.0, 40.8, 34.9, 44.0, 37.4, 50.4, 53.0, 54.0, 29.2, 0.0], [44.0, 40.8, 34.9, 44.3, 37.5, 50.5, 53.0, 53.800000000000004, 28.4, 0.0], [44.3, 40.5, 34.5, 44.2, 37.7, 50.5, 53.1, 54.0, 27.200000000000003, 0.0], [44.6, 40.5, 34.4, 44.0, 37.8, 50.4, 53.2, 53.6, 26.700000000000003, 0.0], [44.7, 40.6, 34.699999999999996, 43.9, 37.8, 50.2, 52.7, 53.6, 25.900000000000002, 0.0], [44.9, 40.699999999999996, 34.1, 44.0, 38.2, 50.0, 52.7, 53.400000000000006, 24.7, 0.0], [44.7, 40.400000000000006, 33.6, 43.9, 37.9, 50.3, 52.6, 53.300000000000004, 24.099999999999998, 0.0], [44.6, 40.400000000000006, 33.7, 43.7, 38.1, 50.2, 52.900000000000006, 52.7, 23.400000000000002, 0.0], [44.6, 40.2, 33.2, 43.7, 38.1, 50.4, 53.0, 52.7, 22.0, 0.0], [44.2, 40.1, 32.800000000000004, 43.7, 38.4, 49.9, 52.800000000000004, 52.400000000000006, 20.8, 0.0], [44.2, 40.1, 32.5, 43.5, 38.2, 49.9, 52.6, 52.2, 19.0, 0.0], [44.3, 40.1, 32.4, 43.3, 38.1, 49.7, 52.400000000000006, 52.400000000000006, 18.0, 0.0], [44.1, 40.1, 32.4, 43.2, 38.0, 49.3, 52.0, 52.400000000000006, 17.299999999999997, 0.0], [43.7, 39.5, 32.7, 42.6, 38.0, 49.1, 51.800000000000004, 52.1, 16.3, 0.0], [43.6, 39.5, 32.5, 43.0, 37.7, 48.699999999999996, 51.4, 51.800000000000004, 15.6, 0.0]],
    ]


def main():
    alpha_grid = np.arange(0, 1.001, 0.02)
    all_accs = get_accs()

    colors = sns.color_palette("husl", 9)
    with sns.axes_style("darkgrid"):
        for i, accs in enumerate(all_accs):
            accs = np.mean(accs, axis=1)
            plt.plot(alpha_grid, accs, label=f'task {i+1}', color=colors[i])

    plt.legend()
    plt.ylabel('Test accuracy')
    plt.xlabel('Interpolation α')
    plt.show()


if __name__ == '__main__':
    main()
