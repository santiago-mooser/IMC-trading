# IMC-trading

This repository is the code used by our team, Quant Crusaders, in IMC's prosperity trading competition.'



# What files to look at for reference

Data analysis: [rounds/round_3/analysis.ipyb](https://github.com/santiago-mooser/IMC-trading/blob/main/rounds/round_3/analysis.ipynb)

Strategy implementation: [rounds/round_3/round_3.py](https://github.com/santiago-mooser/IMC-trading/blob/main/rounds/round_3/round_3.py)

Some writeup: [rounds/round_2/writeup.md](https://github.com/santiago-mooser/IMC-trading/blob/main/rounds/round_2/writeup.md)


# Notes during the challange

Coding info:
https://imc-prosperity.notion.site/Writing-an-Algorithm-in-Python-658e233a26e24510bfccf0b1df647858

Useful stuff:

## Previous events search:
https://duckduckgo.com/?t=ffab&q=site%3Agithub.com+imc+prospoerity+page&ia=web

## Potential market-making strategies:
https://github.com/Andrew-Lloy/Optimal-HFT-Guilbaud-Pham/blob/main/1106.5040.pdf

## Example of previous year's results:
- https://github.com/nicolassinott/IMC_Prosperity/tree/main 
- #2 trading bot in 2023: https://github.com/ShubhamAnandJain/IMC-Prosperity-2023-Stanford-Cardinal

## Trade visualizer (requires custom logging)
https://jmerle.github.io/imc-prosperity-2-visualizer/

## Training a linear regression model
https://www.alpharithms.com/predicting-stock-prices-with-linear-regression-214618/

## Hidden markov chains in python
https://medium.com/@natsunoyuki/hidden-markov-models-with-python-c026f778dfa7

---

# Last year's format:

1. Round 1: Bananas and Pearls
    - Bananas have a volatile price(but relatively linear)
    - Pearls had a very stable price

2. Round 2: Coconuts and Pina Coladas
    - Coconuts are an ingredient in Pina Coladas
    - Thus price of Pina Coladas are derived from the price of Coconuts

3. Round 3: Berries and Diving Gear
    - Berries are a seasonal good
    - Diving Gear is influenced by Dolphin Sightings

4. Round 4: Picnic Baskets, Ukulele, Dip and Baguette
    - A picnic basket has one ukulele, one dip and one Baguette
    - price of picnic basket is not equivalent to the sum of the 3 goods

5. Round 5: Can use any IMC bots or previous bots


General Strat:
    1. For stable goods, derive a strike price for both buying and selling
    2. For volatile goods, develop a base momentum trading model
    3. For related goods, develop a base pair trading model

Specific Strat:
- Bananas: Use a RSI with a wider range specific to bananas (like 80 20)

Source: https://github.com/liamjdavis/IMC-Prosperity-2024/blob/main/plan.txt
