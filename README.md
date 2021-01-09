# Choosing Restaurant using Fuzzy Logic System
Fuzzy system to help us decide whether to go to a restaurant based on its distance, price and rating.

## Usage
| Argument Parameter | Description                                |
|--------------------|--------------------------------------------|
| --graph            | To display graph of membership functions.  |
| --input_csv        | Path to input data csv file.               |
| --output_csv       | Path to save predicted output in csv file. |

### Part 1
1. To run fuzzy system with console input:  
```commandline
python part1.py
```
   
2. To run fuzzy system with input csv file:  
```commandline
python part1.py --input_csv=<path to read input csv> --output_csv=<path to save output csv>
```

3. To display membership function graph:  
```commandline
python part1.py --graph
```
or
```commandline
python part1.py --input_csv=<path to read input csv> --output_csv=<path to save output csv> --graph
```

### Part 2
To run part 2 is the same as part 1 but replace filename `part1.py` with `part2.py`.

## Input CSV File Structure
The input csv file should have header as below:  

| price | distance | rating | label |
|-------|----------|--------|-------|
| data  | data     | data   | data  |