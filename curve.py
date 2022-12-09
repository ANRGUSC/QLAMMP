import numpy as np
import math
from matplotlib import pyplot as plt
from amm import AMM

# we are dealing with integer mathematics everywhere, so this is the epsillon
EPSILLON = 1e-5


class Curve(AMM):
    def __init__(self, reserves: list[int], leverage: float, fee: float):
        n = len(reserves)
        super().__init__(reserves, [1 / n] * n)

        if leverage < 0:
            raise Exception("leverage can go from zero to infinity only")

        self.A = leverage
        self.n = n
        self.fee = fee
        self.fee_collected = 0
        self.numSwaps = 0
        self.totSwaps = 0
        self.vol = 0
        self.feeRewards = 0
        self.currSlippage = None
        # self.minSlippage = np.Inf
        # self.maxSlippage = np.NINF
        # self.feeList = []

    def plusTotSwaps(self):
        self.totSwaps += 1

    def plusFee(self):
        self.fee += 1

    def minusFee(self):
        self.fee -= 1

    def plusLev(self):
        self.A += 2

    def minusLev(self):
        self.A -= 2
    
    def setFee(self, fee):
        assert (fee >= 4 and fee <= 30), "fee out of bounds"
        self.fee = fee
    
    def setLev(self, lev):
        assert (lev>=0 and lev <= 1000), "LevCoeff out of bounds"
        self.A = lev

    def getFee(self):
        return self.fee

    def getLev(self):
        return self.A

    def getFeeCollected(self):
        return self.fee_collected

    def getVolume(self):
        return self.vol
    
    def getFeeRewards(self):
        k = self.feeRewards
        self.feeRewards = 0
        # print(f">>> Env got rewards.. ${k} NumSwaps: {self.totSwaps} Fee%: {self.fee}")
        return k, self.currSlippage, self.totSwaps
    
    # def getNumSwaps(self):
    #     return self.numSwaps

    def _spot_price(self, updated_reserves_in: int, updated_reserves_out: int):
        """Used for implicit divergence loss computation only. This is a modified
        version of spot_price, where we do not update the state of the instance.

        Args:
            updated_reserves_in (int): reserves of in asset with which to compute the
            spot price
            updated_reserves_out (int): reserves of out asset with which to compute the
            spot price

        Returns:
            [float]: new spot price
        """
        C = self._get_sum_invariant()
        X = (C / self.n) ** self.n
        amplified_prod_inv = self.A * X
        ccnn = C * ((C / self.n) ** self.n)
        numerator = updated_reserves_in * (
            amplified_prod_inv * updated_reserves_out + ccnn
        )
        denominator = updated_reserves_out * (
            amplified_prod_inv * updated_reserves_in + ccnn
        )
        return float(numerator) / denominator

    def spot_price(self, asset_in_ix: int, asset_out_ix: int):
        C = self._get_sum_invariant()
        X = (C / self.n) ** self.n
        amplified_prod_inv = self.A * X
        ccnn = C * ((C / self.n) ** self.n)
        numerator = self.reserves[asset_in_ix] * (
            amplified_prod_inv * self.reserves[asset_out_ix] + ccnn
        )
        denominator = self.reserves[asset_out_ix] * (
            amplified_prod_inv * self.reserves[asset_in_ix] + ccnn
        )
        return float(numerator) / denominator

    def _compute_trade_qty_out(
        self, qty_in: int, asset_in_ix: int, asset_out_ix: int, collect: bool
    ):
        C = self._get_sum_invariant()
        X = (C / self.n) ** self.n

        if collect:
            self.fee_collected += qty_in * self.fee/10000
            self.feeRewards += qty_in * self.fee/10000
        qty_in -= qty_in * self.fee/10000

        updated_reserves_in = self.reserves[asset_in_ix] + qty_in
        # new pool product excluding output asset
        prod_exo = (
            math.prod(self.reserves)
            / (self.reserves[asset_in_ix] * self.reserves[asset_out_ix])
            * updated_reserves_in
        )

        # new pool sum excluding output asset
        sum_exo = sum(self.reserves) + qty_in - self.reserves[asset_out_ix]

        # + EPSILLON everywhere here to avoid division by zero
        A = max(self.A, EPSILLON)
        B = (1 - 1 / A) * C - sum_exo
        updated_reserves_out = (
            B + math.sqrt((B ** 2 + 4 * C * X / A / prod_exo))
        ) / 2

        return updated_reserves_in, updated_reserves_out

    def trade(self, qty_in: int, asset_in_ix: int, asset_out_ix: int):
        (
            updated_reserves_in,
            updated_reserves_out,
        ) = self._compute_trade_qty_out(qty_in, asset_in_ix, asset_out_ix, True)

        prev_reserves_out = self.reserves[asset_out_ix]
        self.reserves[asset_in_ix] = updated_reserves_in
        self.reserves[asset_out_ix] = updated_reserves_out
        self.numSwaps += 1
        self.vol += qty_in
        # self.feeList.append(self.fee)
        # self.fee = min(30, self.fee + 0.75)

        # print(f"Trade Complete... Reserves: {self.reserves} Fee Collected: {round(self.fee_collected, 4)} Volume Traded: {self.vol} #Swaps: {self.numSwaps}")

        return prev_reserves_out - updated_reserves_out

    def getSlippage(self, qty_in: int, asset_in_ix: int, asset_out_ix: int):
        (
            updated_reserves_in,
            updated_reserves_out,
        ) = self._compute_trade_qty_out(qty_in, asset_in_ix, asset_out_ix, False)

        prev_reserves_out = self.reserves[asset_out_ix]

        self.currSlippage = ((qty_in - (prev_reserves_out - updated_reserves_out)) / qty_in) * 100
        # self.minSlippage = min(self.minSlippage, self.currSlippage)
        # self.maxSlippage = max(self.maxSlippage, self.currSlippage)

        return self.currSlippage

    def slippage(self, qty_in: int, asset_in_ix: int, asset_out_ix: int):
        _, updated_reserves_out_ix = self._compute_trade_qty_out(
            qty_in, asset_in_ix, asset_out_ix, False
        )
        x_2 = self.reserves[asset_out_ix] - updated_reserves_out_ix
        x_1 = qty_in
        p = self._spot_price(
            self.reserves[asset_in_ix], self.reserves[asset_out_ix]
        )
        return ((x_1 / x_2) / p) - 1  # NOTE: Effective exchange rate / Pre Swap Quote - 1

    # ! notice that the signature here is different to the one in AMM
    # ! this one is missing pct_change. Rather it is computed implicitly.
    def divergence_loss(
        self, qty_in: int, asset_in_ix: int, asset_out_ix: int
    ):
        # for different quantities of asset_in_ix, figure out what is the percentage change
        # then plot divergence loss versus this percentage change

        # todo: this is the same as in AMM. This is not DRY
        if qty_in < 0:
            if self.reserves[asset_in_ix] < -qty_in:
                raise Exception("invalid quantity to deplete")

        # NOTE: pct_change is the percentage change in the spot price

        pre_trade_spot_price = self._spot_price(
            self.reserves[asset_in_ix], self.reserves[asset_out_ix]
        )
        (
            updated_reserves_in_ix,
            updated_reserves_out_ix,
        ) = self._compute_trade_qty_out(qty_in, asset_in_ix, asset_out_ix, True)
        post_trade_spot_price = self._spot_price(
            updated_reserves_in_ix, updated_reserves_out_ix
        )
        pct_change = post_trade_spot_price / pre_trade_spot_price - 1

        # now return divergence loss and pct_change
        value_pool = (
            updated_reserves_in_ix
            + updated_reserves_out_ix * post_trade_spot_price
        )

        return (
            pct_change,
            value_pool / self.value_hold(pct_change, asset_in_ix, asset_out_ix)
            - 1,
        )

    def _get_sum_invariant(self):
        sum_all = sum(self.reserves)
        product_all = math.prod(self.reserves)

        # Special case with qual size pool, no need to calculate, although results are the same
        if len(set(self.reserves)) == 1:
            return sum_all

        # Special case with a=0 or 1, no need to calculate, although results are the same
        if self.A < 1e-10:
            return product_all ** (1 / self.n) * self.n

        if self.A == 1:
            return (product_all * sum_all) ** (1 / (self.n + 1)) * self.n ** (
                self.n / (self.n + 1)
            )

        if self.n == 2:
            sqrtand = (
                product_all
                * (
                    9 * self.A * sum_all
                    + math.sqrt(
                        81 * self.A ** 2 * sum_all ** 2
                        + 48 * product_all * (self.A - 1) ** 3
                    )
                )
            ) ** (1 / 3)
            suminv_complex = (
                -2 * 6 ** (2 / 3) * product_all * (self.A - 1)
                + 6 ** (1 / 3) * sqrtand ** 2
            ) / (3 * sqrtand)
            return suminv_complex.real

        raise Exception("cannot handle unequal asset pool with n>2")

    def getStats(self):
        print("=======================================================================================")
        print(f"Final Stats: Reserves: {self.reserves} Fee Collected: {round(self.fee_collected, 4)}\nVolume Traded: {self.vol} #Swaps: {self.numSwaps} #Cancelled: {400 - self.numSwaps}\nAvg Fee%: {(self.fee_collected/self.vol) * 100}%\nTotal Slippage: {((sum(self.reserves) + self.fee_collected - 40000)/self.vol) * 100}%")

        # plt.plot(self.feeList)
        # plt.show()


if __name__ == "__main__":
    curve = Curve([1_000, 2_000], 10, 15)

    # when leverage is zero, we are reducing to constant sum
    _qty_in = np.arange(-950, 2_000, 1_00)
    pct_changes = []
    divergence_loss = []

    for qty in _qty_in:
        (x, y) = curve.divergence_loss(qty, 0, 1)
        pct_changes.append(x)
        divergence_loss.append(y)

    print(curve.fee_collected)
    plt.plot(pct_changes, divergence_loss)
    plt.show()