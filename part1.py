import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_mf
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--graph", default=False, help='Display graph', action='store_true')
parser.add_argument("--input_csv", default='', type=str, help='Input csv file path')
parser.add_argument("--output_csv", default='./output_part1.csv', type=str, help='Input csv file path')
args = parser.parse_args()

# Define universe
price = np.arange(0, 31, 1)
distance = np.arange(0, 21, 1)
rating = np.arange(0, 6, 1)
going = 1
not_going = 0
label = np.arange(0, 1.01, 0.01)

# Define membership function
cheap = fuzz.trapmf(price, [0, 0, 7, 10])
medium_price = fuzz.trimf(price, [7, 12, 18])
expensive = fuzz.trapmf(price, [14, 25, 30, 30])

close = fuzz.trapmf(distance, [0, 0, 5, 7])
medium_distance = fuzz.trimf(distance, [5, 10, 15])
far = fuzz.trapmf(distance, [10, 15, 20, 20])

bad = fuzz.trapmf(rating, [0, 0, 2, 3])
moderate = fuzz.trimf(rating, [2, 3, 4])
good = fuzz.trapmf(rating, [3, 4, 5, 5])

not_go = fuzz.trimf(label, [0, 0, 1.0])
will_go = fuzz.trimf(label, [0, 1.0, 1.0])

# Plot MF graph
if args.graph:
    plot_mf(price, low=cheap, medium=medium_price, high=expensive, xlabel='price (RM)', ylabel='membership degree',
            title='The food price (RM)', legends=['cheap', 'medium', 'expensive'])
    plot_mf(distance, low=close, medium=medium_distance, high=far, xlabel='distance (KM)', ylabel='membership degree',
            title='Restaurant distance (KM)', legends=['close', 'medium', 'far'])
    plot_mf(rating, low=bad, medium=moderate, high=good, xlabel='rating', ylabel='membership degree',
            title='Ovarall rating', legends=['bad', 'moderate', 'good'])
    plot_mf(label, low=not_go, high=will_go, xlabel='going', ylabel='membership degree',
            title='Possibility of going to the restaurant', legends=['not going', 'going'])


def predict(price_input, distance_input, rating_input):
    # R1: If the price is cheap, distance is close, rating is good, then consider going.
    r1_x1 = fuzz.interp_membership(price, cheap, price_input)
    r1_x2 = fuzz.interp_membership(distance, close, distance_input)
    r1_x3 = fuzz.interp_membership(rating, good, rating_input)
    fire_r1 = min(r1_x1, r1_x2, r1_x3)
    # R2: If the price is cheap, distance is medium, rating is good, then consider going.
    r2_x1 = fuzz.interp_membership(price, cheap, price_input)
    r2_x2 = fuzz.interp_membership(distance, medium_distance, distance_input)
    r2_x3 = fuzz.interp_membership(rating, good, rating_input)
    fire_r2 = min(r2_x1, r2_x2, r2_x3)
    # R3: if the price is expensive, distance is far, rating is bad, then consider not going.
    r3_x1 = fuzz.interp_membership(price, expensive, price_input)
    r3_x2 = fuzz.interp_membership(distance, far, distance_input)
    r3_x3 = fuzz.interp_membership(rating, bad, rating_input)
    fire_r3 = min(r3_x1, r3_x2, r3_x3)
    # R4: If the price is expensive, distance is far, rating is moderate, then consider not going.
    r4_x1 = fuzz.interp_membership(price, expensive, price_input)
    r4_x2 = fuzz.interp_membership(distance, far, distance_input)
    r4_x3 = fuzz.interp_membership(rating, moderate, rating_input)
    fire_r4 = min(r4_x1, r4_x2, r4_x3)
    # R5: If the price is expensive, distance is medium, rating is bad, then consider not going.
    r5_x1 = fuzz.interp_membership(price, expensive, price_input)
    r5_x2 = fuzz.interp_membership(distance, medium_distance, distance_input)
    r5_x3 = fuzz.interp_membership(rating, bad, rating_input)
    fire_r5 = min(r5_x1, r5_x2, r5_x3)
    # R6: If the price is medium, distance is close, rating is good, then consider going.
    r6_x1 = fuzz.interp_membership(price, medium_price, price_input)
    r6_x2 = fuzz.interp_membership(distance, close, distance_input)
    r6_x3 = fuzz.interp_membership(rating, good, rating_input)
    fire_r6 = min(r6_x1, r6_x2, r6_x3)
    # R7: If the price is medium, distance is far, rating is bad, then consider not going.
    r7_x1 = fuzz.interp_membership(price, medium_price, price_input)
    r7_x2 = fuzz.interp_membership(distance, far, distance_input)
    r7_x3 = fuzz.interp_membership(rating, bad, rating_input)
    fire_r7 = min(r7_x1, r7_x2, r7_x3)
    # R8: If the price is medium, distance is far, rating is moderate, then consider not going.
    r8_x1 = fuzz.interp_membership(price, medium_price, price_input)
    r8_x2 = fuzz.interp_membership(distance, far, distance_input)
    r8_x3 = fuzz.interp_membership(rating, moderate, rating_input)
    fire_r8 = min(r8_x1, r8_x2, r8_x3)
    # R9: If the price is medium, distance is medium, rating is bad, then consider not going.
    r9_x1 = fuzz.interp_membership(price, medium_price, price_input)
    r9_x2 = fuzz.interp_membership(distance, medium_distance, distance_input)
    r9_x3 = fuzz.interp_membership(rating, bad, rating_input)
    fire_r9 = min(r9_x1, r9_x2, r9_x3)
    # R10: If the price is medium, distance is medium, rating is good, then consider going.
    r10_x1 = fuzz.interp_membership(price, medium_price, price_input)
    r10_x2 = fuzz.interp_membership(distance, medium_distance, distance_input)
    r10_x3 = fuzz.interp_membership(rating, good, rating_input)
    fire_r10 = min(r10_x1, r10_x2, r10_x3)

    rule_1_clip = np.fmin(fire_r1, will_go)
    rule_2_clip = np.fmin(fire_r2, will_go)
    rule_3_clip = np.fmin(fire_r3, not_go)
    rule_4_clip = np.fmin(fire_r4, not_go)
    rule_5_clip = np.fmin(fire_r5, not_go)
    rule_6_clip = np.fmin(fire_r6, will_go)
    rule_7_clip = np.fmin(fire_r7, not_go)
    rule_8_clip = np.fmin(fire_r8, not_go)
    rule_9_clip = np.fmin(fire_r9, not_go)
    rule_10_clip = np.fmin(fire_r10, will_go)

    # Aggregate all rules
    temp1 = np.fmax(rule_1_clip, rule_2_clip)
    temp2 = np.fmax(temp1, rule_3_clip)
    temp3 = np.fmax(temp2, rule_4_clip)
    temp4 = np.fmax(temp3, rule_5_clip)
    temp5 = np.fmax(temp4, rule_6_clip)
    temp6 = np.fmax(temp5, rule_7_clip)
    temp7 = np.fmax(temp6, rule_8_clip)
    temp8 = np.fmax(temp7, rule_9_clip)
    output = np.fmax(temp8, rule_10_clip)

    # Defuzzification
    try:
        going_predict = fuzz.defuzz(label, output, 'centroid')
        if args.graph:
            fire_going = fuzz.interp_membership(label, output, going_predict)
            label_0 = np.zeros_like(label)
            fig, ax0 = plt.subplots(figsize=(8, 3))
            ax0.plot(label, will_go, 'g', linestyle='--')
            ax0.plot(label, not_go, 'r', linestyle='--')
            ax0.fill_between(label, label_0, output, facecolor='Orange', alpha=0.5)
            ax0.plot([going_predict, going_predict], [0, fire_going], 'k', linewidth=2.5, alpha=0.9)
            ax0.get_xaxis().tick_bottom()
            ax0.get_yaxis().tick_left()
            ax0.set_xlim([min(label), max(label)])
            ax0.set_ylim([0, 1])
            plt.xlabel('Possibility of Going to the Restaurant')
            plt.ylabel('membership degree')
            plt.title('Restaurant ')
            plt.show()
        return going_predict
    except AssertionError:
        return -1


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
            # Getting user inputs
            price_input = input('Enter price (RM):')
            distance_input = input('Enter distance (KM):')
            rating_input = input('Enter rating (1 - 5):')
            try:
                going_predict = predict(price_input, distance_input, rating_input)
                print('The possibility of going is: ', round(going_predict, 4))
                print('Predicted label:', 'going' if going_predict >= 0.5 else 'not going')
            except AssertionError:
                print('zero error')
