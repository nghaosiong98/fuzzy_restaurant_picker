import skfuzzy as fuzz
import numpy as np
from utils import plot_mf

# Define universe
distance = np.arange(0, 21, 1)
price = np.arange(0, 31, 1)
rating = np.arange(0, 5.1, 0.1)
going = np.arange(0, 1.01, 0.01)

# Define membership function
near = fuzz.trapmf(distance, [0, 0, 5, 7])
middle_dist = fuzz.trapmf(distance, [5, 7, 10, 12])
far = fuzz.trapmf(distance, [10, 12, 20, 20])

cheap = fuzz.trimf(price, [0, 0, 11])
middle_price = fuzz.trimf(price, [8, 12, 16])
expensive = fuzz.trapmf(price, [12, 20, 30, 30])

bad = fuzz.trimf(rating, [0, 0, 2.2])
medium = fuzz.trimf(rating, [1.5, 2.5, 3.5])
good = fuzz.trimf(rating, [2.8, 5, 5])

not_go = fuzz.trimf(going, [0, 0, 0.75])
will_go = fuzz.trimf(going, [0.25, 1.0, 1.0])

plot_mf(distance, low=near, medium=middle_dist, high=far, xlabel='distance (KM)', ylabel='membership degree')
plot_mf(price, low=cheap, medium=middle_price, high=expensive, xlabel='price (RM)', ylabel='membership degree')
plot_mf(rating, low=bad, medium=medium, high=good, xlabel='rating', ylabel='membership degree')
plot_mf(going, low=not_go, high=going, xlabel='going', ylabel='membership degree')

distance_input = 8
price_input = 10
rating_input = 4.2

# Rule 1: The restaurant is far, price is cheap
rule_1_a = fuzz.interp_membership(distance, far, distance_input)
rule_1_b = fuzz.interp_membership(price, cheap, price_input)
fire_rule_1 = min(rule_1_a, rule_1_b)
# Rule 2: The restaurant is far, rating is good
rule_2_a = fuzz.interp_membership(distance, far, distance_input)
rule_2_b = fuzz.interp_membership(rating, good, rating_input)
fire_rule_2 = min(rule_2_a, rule_2_b)
# Rule 3: The restaurant is close, price is expensive
rule_3_a = fuzz.interp_membership(distance, near, distance_input)
rule_3_b = fuzz.interp_membership(price, expensive, price_input)
fire_rule_3 = min(rule_3_a, rule_3_b)
# Rule 4: The restaurant is close, rating is bad, price is cheap
rule_4_a = fuzz.interp_membership(distance, near, distance_input)
rule_4_b = fuzz.interp_membership(rating, bad, rating_input)
rule_4_c = fuzz.interp_membership(price, cheap, price_input)
fire_rule_4 = min(rule_4_a, rule_4_b, rule_4_c)
# Rule 5: The restaurant is close.
rule_5 = fuzz.interp_membership(distance, near, distance_input)
fire_rule_5 = rule_5
# Rule 6: The price is medium.
rule_6 = fuzz.interp_membership(price, middle_price, price_input)
fire_rule_6 = rule_6
# Rule 7: The restaurant is far, but rating is good and price is cheap
rule_7_a = fuzz.interp_membership(distance, far, distance_input)
rule_7_b = fuzz.interp_membership(rating, good, rating_input)
rule_7_c = fuzz.interp_membership(price, cheap, price_input)
fire_rule_7 = min(rule_7_a, rule_7_b, rule_7_c)
# Rule 8: The restaurant distance is medium, rating is bad, price is expensive
rule_8_a = fuzz.interp_membership(distance, middle_dist, distance_input)
rule_8_b = fuzz.interp_membership(rating, bad, rating_input)
rule_8_c = fuzz.interp_membership(price, expensive, price_input)
fire_rule_8 = min(rule_8_a, rule_8_b, rule_8_c)
# Rule 9: The price is expensive.
rule_9 = fuzz.interp_membership(price, expensive, price_input)
fire_rule_9 = rule_9
# Rule 10: The rating is good, price is cheap
# rule_10 = fuzz.interp_membership()

rule_1_clip = np.fmin(fire_rule_1, will_go)
rule_2_clip = np.fmin(fire_rule_2, will_go)
rule_3_clip = np.fmin(fire_rule_3, not_go)
rule_4_clip = np.fmin(fire_rule_4, not_go)
rule_5_clip = np.fmin(fire_rule_5, will_go)
rule_6_clip = np.fmin(fire_rule_6, will_go)
rule_7_clip = np.fmin(fire_rule_7, will_go)
rule_8_clip = np.fmin(fire_rule_8, not_go)
rule_9_clip = np.fmin(fire_rule_9, not_go)

# Aggregate all rules
temp1 = np.fmax(rule_1_clip, rule_2_clip)
temp2 = np.fmax(temp1, rule_3_clip)
temp3 = np.fmax(temp2, rule_4_clip)
temp4 = np.fmax(temp3, rule_5_clip)
temp5 = np.fmax(temp4, rule_6_clip)
temp6 = np.fmax(temp5, rule_7_clip)
temp7 = np.fmax(temp6, rule_8_clip)
output = np.fmax(temp7, rule_9_clip)

# Defuzzification
going_predict = fuzz.defuzz(going, output, 'centroid')
print('The possibility of going is: ', going_predict)

# fire_going = fuzz.interp_membership(going, output, going_predict)
# going_0 = np.zeros_like(going)
# fig, ax0 = plt.subplots(figsize=(8, 3))
# ax0.plot(going, will_go, 'g', linestyle='--')
# ax0.plot(going, not_go, 'r', linestyle='--')
# ax0.fill_between(going, going_0, output, facecolor='Orange', alpha=0.5)
# ax0.plot([going_predict, going_predict], [0, fire_going], 'k', linewidth=2.5, alpha=0.9)
# ax0.get_xaxis().tick_bottom()
# ax0.get_yaxis().tick_left()
# ax0.set_xlim([min(going), max(going)])
# ax0.set_ylim([0, 1])
# plt.xlabel('Possibility of Going to the Restaurant')
# plt.ylabel('membership degree')
# plt.title('Restaurant ')
# plt.show()
