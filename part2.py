import numpy as np
import skfuzzy as fuzz
from utils import plot_mf
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--graph", default=False, help='Display graph', action='store_true')
parser.add_argument("--input_csv", default='', type=str, help='Input csv file path')
parser.add_argument("--output_csv", default='./output_part2.csv', type=str, help='Input csv file path')
args = parser.parse_args()

# Universe
price = np.arange(0, 31, 1)
distance = np.arange(0, 20.1, 0.1)
rating = np.arange(0, 6, 1)
label = np.arange(0, 1.01, 0.01)
going = 1
not_going = 0

# Membership functions
cheap = fuzz.trapmf(price, [0, 0, 6, 18])
medium_price = fuzz.trimf(price, [6, 18, 30])
expensive = fuzz.trapmf(price, [18, 30, 30, 30])

close = fuzz.trapmf(distance, [0, 0, 5, 12.5])
medium_distance = fuzz.trimf(distance, [5, 12.5, 20])
far = fuzz.trapmf(distance, [12.5, 20, 20, 20])

bad = fuzz.trapmf(rating, [0, 0, 2, 3])
moderate = fuzz.trimf(rating, [2, 3, 4])
good = fuzz.trapmf(rating, [3, 4, 5, 5])

# Plot MF graph
if args.graph:
    plot_mf(price, low=cheap, medium=medium_price, high=expensive, xlabel='price (RM)', ylabel='membership degree',
            title='The food price (RM)', legends=['cheap', 'medium', 'expensive'])
    plot_mf(distance, low=close, medium=medium_distance, high=far, xlabel='distance (KM)', ylabel='membership degree',
            title='Restaurant distance (KM)', legends=['close', 'medium', 'far'])
    plot_mf(rating, low=bad, medium=moderate, high=good, xlabel='rating', ylabel='membership degree',
            title='Ovarall rating', legends=['bad', 'moderate', 'good'])


def predict(price_input, distance_input, rating_input):
    # Inference
    # R1: If price is cheap, distance is close and rating is good, then consider going.
    r1_x1 = fuzz.interp_membership(price, cheap, price_input)
    r1_x2 = fuzz.interp_membership(distance, close, distance_input)
    r1_x3 = fuzz.interp_membership(rating, good, rating_input)
    fire_r1 = r1_x1 * r1_x2 * r1_x3
    # R2: If price is cheap, distance is far and rating is bad, then consider not going.
    r2_x1 = fuzz.interp_membership(price, cheap, price_input)
    r2_x2 = fuzz.interp_membership(distance, far, distance_input)
    r2_x3 = fuzz.interp_membership(rating, bad, rating_input)
    fire_r2 = r2_x1 * r2_x2 * r2_x3
    # R3: If price is cheap, distance is medium and rating is good, then consider going.
    r3_x1 = fuzz.interp_membership(price, cheap, price_input)
    r3_x2 = fuzz.interp_membership(distance, medium_distance, distance_input)
    r3_x3 = fuzz.interp_membership(rating, good, rating_input)
    fire_r3 = r3_x1 * r3_x2 * r3_x3
    # R4: If price is cheap, distance is medium, and rating is moderate, then consider going.
    r4_x1 = fuzz.interp_membership(price, cheap, price_input)
    r4_x2 = fuzz.interp_membership(distance, medium_distance, distance_input)
    r4_x3 = fuzz.interp_membership(rating, moderate, rating_input)
    fire_r4 = r4_x1 * r4_x2 * r4_x3
    # R5: If price is cheap or medium, distance is far, and rating is moderate, then consider not going.
    r5_x1_cheap = fuzz.interp_membership(price, cheap, price_input)
    r5_x1_medium = fuzz.interp_membership(price, medium_price, price_input)
    r5_x1 = max(r5_x1_cheap, r5_x1_medium)
    r5_x2 = fuzz.interp_membership(distance, medium_distance, distance_input)
    r5_x3 = fuzz.interp_membership(rating, moderate, rating_input)
    fire_r5 = r5_x1 * r5_x2 * r5_x3
    # R6: If price is expensive, distance is far, and rating is bad, then consider not going.
    r6_x1 = fuzz.interp_membership(price, expensive, price_input)
    r6_x2 = fuzz.interp_membership(distance, far, distance_input)
    r6_x3 = fuzz.interp_membership(rating, bad, rating_input)
    fire_r6 = r6_x1 * r6_x2 * r6_x3
    # R7: If price is expensive, distance is medium, and rating is bad, then consider not going.
    r7_x1 = fuzz.interp_membership(price, expensive, price_input)
    r7_x2 = fuzz.interp_membership(distance, medium_distance, distance_input)
    r7_x3 = fuzz.interp_membership(rating, bad, rating_input)
    fire_r7 = r7_x1 * r7_x2 * r7_x3
    # R8: If price is medium, distance is medium, and rating is bad, then consider not going.
    r8_x1 = fuzz.interp_membership(price, medium_price, price_input)
    r8_x2 = fuzz.interp_membership(distance, medium_distance, distance_input)
    r8_x3 = fuzz.interp_membership(rating, bad, rating_input)
    fire_r8 = r8_x1 * r8_x2 * r8_x3
    # R9: If price is medium, distance is medium, and rating is moderate, then consider not going.
    r9_x1 = fuzz.interp_membership(price, medium_price, price_input)
    r9_x2 = fuzz.interp_membership(distance, medium_distance, distance_input)
    r9_x3 = fuzz.interp_membership(rating, moderate, rating_input)
    fire_r9 = r9_x1 * r9_x2 * r9_x3

    # Defuzzification
    sum_of_fire = fire_r1 + fire_r2 + fire_r3 + fire_r4 + fire_r5 + fire_r6 + fire_r7 + fire_r8 + fire_r9

    output = 0 if sum_of_fire == 0 else (
                                                fire_r1 * going + fire_r2 * not_going + fire_r3 * going + fire_r4 * going
                                                + fire_r5 * not_going + fire_r6 * not_going + fire_r7 * not_going
                                                + fire_r8 * not_going + fire_r9 * not_going
                                        ) / sum_of_fire
    return output


if __name__ == '__main__':
    if args.input_csv is not '':
        # Read input from csv file
        dataset = pd.read_csv(args.input_csv)
        dataset['predicted_possibility'] = dataset.apply(
            lambda row: predict(row['price'], row['distance'], row['rating']),
            axis=1)
        dataset['predicted_label'] = dataset.apply(
            lambda row: 'zero error' if row['predicted_possibility'] == -1 else (
                'going' if row['predicted_possibility'] >= 0.5 else 'not going'),
            axis=1)
        dataset.to_csv(args.output_csv, index=False)
    else:
        while True:
            # Get user input
            price_input = input('Enter price (RM):')
            distance_input = input('Enter distance (KM):')
            rating_input = input('Enter rating (0.0 - 5.0):')
            output = predict(price_input, distance_input, rating_input)
            print('You can consider', 'going' if output >= 0.5 else 'not going', 'to this restaurant.')
