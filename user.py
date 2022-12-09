import numpy as np
import random
from colorama import Fore, Style

class User():
    def __init__(self, init_t0, init_t1, userNum):
        self.bal = [init_t0, init_t1]
        self.tolerance = None   # 0.1% to 5%
        self.urgency = None     # ln(1) to ln(2)
        self.userNum = userNum

    def _updateUrgency(self):
        if self.urgency < np.log(2):
            self.urgency += self.urgency * 0.1

    def _swap(self, idx_in, amount, Curve):

        """
        Used by user for swapping tokens using the protocol.

        Args:
            idx_in (int): index of token to sell to the protocol
            amount (float): amount of token[idx_in] to be swapped
            Curve (AMM): protocol to trade with

        Returns:
            [float]: amount of paired token received
        """

        assert (idx_in == 0 or idx_in == 1), "idx_in out of bounds"
        assert (amount > 0 and amount <= self.bal[idx_in]), f"amount out of bounds, {amount}, {self.bal[idx_in]}"
        
        slippage = Curve.getSlippage(amount, idx_in, 1 - idx_in)
        # Threshold -- tolerance * exponent of urgency
        if slippage < (self.tolerance * np.exp(self.urgency)):
            received = Curve.trade(amount, idx_in, 1 - idx_in)
            self.bal[idx_in] -= amount
            self.bal[1 - idx_in] += received
            # print(f"{Fore.GREEN}User# {self.userNum}{Style.RESET_ALL}: Swap success... Slippage: {round(((amount - received )/ amount) * 100, 4)}%")
            # Curve.plusTotSwaps()
            return 1, received, idx_in, self.tolerance, self.urgency
        else:
            # print(f"Slippage: {slippage} Current Tolerance: {self.tolerance * np.exp(self.urgency)}")
            # print(f"{Fore.BLUE}User# {self.userNum}{Style.RESET_ALL}: Slippage too high - update and wait...")

            # Upto a 10% chance of cancelling the swap based on urgency
            ep = random.random()
            if ep <= 0.4 * (1 - (self.urgency / np.log(2))): 
                # print(f"{Fore.RED}User# {self.userNum}{Style.RESET_ALL}: Swap Cancelled")
                # currFee = Curve.getFee()/2
                # Curve.setFee(max(4, currFee))
                return -1, amount, idx_in, self.tolerance, self.urgency
            
            # If not canceled, update and return
            self._updateUrgency()
            return 0, amount, idx_in, self.tolerance, self.urgency
            
    
    def makeSwap(self, Curve, tolerance, urgency, new, amount = None, index = None):     
        self.tolerance = tolerance
        self.urgency = urgency
        if new:       
            # Chose random index and coin
            idx = random.randint(0, 1)
            # If bal[idx] is less than 20% of paired token then use other token
            if self.bal[idx] < 0.2 * self.bal[1 - idx]:
                idx = 1 - idx 
            amt = random.uniform(1, self.bal[idx] - 1)
            Curve.plusTotSwaps()
        else:
            idx = index
            if amount < self.bal[idx]:
                amt = amount
            else:
                if self.bal[idx] < 0.2 * self.bal[1 - idx]:
                    idx = 1 - idx 
                amt = random.uniform(1, self.bal[idx] - 1)
        return self._swap(idx, amt, Curve)

            