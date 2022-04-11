import pandas as pd
import math
import click


class portfolio:

    def __init__(self, time, close, signal, ratio=0.01, cash=10000):
        self.time = time
        self.close = close
        self.signal = signal
        self.ratio = ratio
        self.cash = cash
        self.origin_cash = cash
        self.coin_nb = 0
        self.coin_cost = math.inf
        self.cost = cash
        self.income = 0
        self.rate_of_return = 0

    def buy(self, i):
        self.coin_nb += 1
        if self.coin_nb == 1:
            self.coin_cost = self.close[i]
        else:
            self.coin_cost = (self.coin_cost + self.close[i]) / self.coin_nb

        self.cash -= self.close[i]

        print('Buy 1 Coin at {} with the price of {}'.format(self.time[i], self.close[i]))

    def sell(self, i):
        self.cash += self.close[i] * self.coin_nb
        self.coin_cost = math.inf
        print('Sell {} Coin(s) at {} with the price of {}'.format(self.coin_nb, self.time[i], self.close[i]))
        self.coin_nb = 0

    def trade(self):
        print("\n=====Transaction=====\n")
        print("From {} to {}:".format(self.time[0], self.time[-1]))
        for i in range(len(self.time)):
            if self.signal[i] == 0 and self.close[i] < self.coin_cost and self.cash > self.close[i]:
                self.buy(i)
            elif self.signal[i] == 1 and self.close[i] > self.coin_cost:
                self.sell(i)
            elif self.close[i] >= self.coin_cost * (1 + self.ratio):
                self.sell(i)
            else:
                pass

    def summary(self):
        if self.coin_nb > 0:
            self.income = self.coin_nb*self.close[-1] + self.cash

        else:
            self.income = self.cash

        self.rate_of_return = (self.income-self.cost)/self.cost
        print('\n=====Summary=====\n')
        print("From {} to {}:".format(self.time[0], self.time[-1]))
        print("Cash ${:.2f}, Coins {}".format(self.cash, self.coin_nb))
        print("Cost ${:.2f}, Income ${:.2f}, Rate of income {:.3%}".format(self.cost, self.income, self.rate_of_return))


@click.command()
@click.option('-model_output', default='./temp/ETH_USD_result.csv', help='file path for GAF-CNN model output csv')
@click.option('-cash', default=10000, help='cost')
@click.option('-ratio', default=0.01, help='threshold ratio to sell no matter signals')

def run(model_output, cash, ratio):
    df = pd.read_csv(model_output)
    predict = list(df.predict)
    close = list(df.close)
    time = list(df.time)

    # cash = 10000 ratio = 0.01
    eth_port = portfolio(time, close, predict, ratio, cash)
    eth_port.trade()
    eth_port.summary()

if __name__ == "__main__":
    run()