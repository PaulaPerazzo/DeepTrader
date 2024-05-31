import math
from collections import namedtuple

import numpy as np
import pandas as pd

Reward = namedtuple('Reward', ('total', 'long', 'short'))

EPS = 1e-20


class DataGenerator():
    def __init__(self,
                 assets_data,
                 rtns_data,
                 market_data,
                 in_features,
                 val_idx,
                 test_idx,
                 batch_size,
                 val_mode=False,
                 test_mode=False,
                 max_steps=12,
                 norm_type='div-last',
                 window_len=20,
                 trade_len=7,
                 mode='train',
                 allow_short=True,
                 ):

        self.assets_features = in_features[0]
        self.market_features = in_features[1]
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.batch_size = batch_size
        self.val_mode = val_mode
        self.test_mode = test_mode
        self.max_steps = max_steps
        self.norm_type = norm_type
        self.window_len = window_len
        self.trade_len = trade_len
        self.mode = mode
        self.allow_short = allow_short

        self.__org_assets_data = assets_data.copy()[:, :, :in_features[0]]
        self.__ror_data = rtns_data
        # ==== input type ====
        self.__assets_data = self.__org_assets_data.copy()

        if allow_short:
            self.__org_market_data = market_data[:, :in_features[1]]
            self.__market_data = self.__org_market_data.copy()

        self.train_set_len = self.val_idx - 2 * self.trade_len - window_len + 1

        # self.order_set = np.arange(2483, self.val_idx - 2 * trade_len)
        self.order_set = np.arange(5 * (self.window_len + 1) - 1, self.val_idx - 6 * trade_len)
        self.tmp_order = np.array([])

        # ==== Sample ====
        self.__sample_p = np.arange(1, len(self.order_set) + 1) / len(self.order_set)
        self.__sample_p = self.__sample_p / sum(self.__sample_p)

        self.step_cnt = 0

    def _step(self):

        if self.test_mode:
            if self.cursor + self.trade_len >= self.__assets_data.shape[1] - 1:
                return None, None, None, None, None, None, None, True
        obs, market_obs, future_return, past_return = self._get_data()
        obs_masks, future_return_masks = self._get_masks(obs, future_return)
        trade_masks = np.logical_or(obs_masks, future_return_masks)
        obs = self._fillna(obs, obs_masks)
        obs_normed = self.__normalize_assets(obs, obs_masks)
        if self.allow_short:
            market_obs_normed = self.__normalize_market(market_obs)
        else:
            market_obs_normed = None

        future_ror = np.prod((future_return + 1), axis=-1)
        future_ror[future_return_masks] = 0.
        future_p = self.__get_p(future_return)

        if np.isnan(obs).any():
            print('still have nan not been filled in obs')
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        elif np.isnan(obs_normed).any():
            print('still have nan not been filled in obs_normed')
            obs_normed = np.nan_to_num(obs_normed, nan=0.0, posinf=0.0, neginf=0.0)

        assert not np.isnan(obs + obs_normed).any()

        self.cursor += self.trade_len
        if self.val_mode:
            done = (self.cursor >= self.test_idx)
        elif self.test_mode:
            done = (self.cursor >= self.__assets_data.shape[1] - 1)
        else:
            done = ((self.cursor >= self.val_idx).any() or (self.step_cnt >= self.max_steps))
        self.step_cnt += 1

        obs, obs_normed = obs.astype(np.float32), obs_normed.astype(np.float32)
        if self.allow_short:
            market_obs = market_obs.astype(np.float32)
            market_obs_normed = market_obs_normed.astype(np.float32)
        future_ror = future_ror.astype(np.float32)
        future_p = future_p.astype(np.float32)

        return obs, obs_normed, market_obs, market_obs_normed, future_ror, future_p, trade_masks, done


    def reset(self, start_point=None):
        """
        :param start_point:
        :return: obs:(batch, num_assets, window_len, in_features)
        """
        self.step_cnt = 1
        if start_point is not None:
            self.cursor = np.array([start_point])
        elif self.val_mode:
            self.cursor = np.array([self.val_idx])
        elif self.test_mode:
            self.cursor = np.array([self.test_idx])
        else:
            if len(self.tmp_order) == 0:
                # self.tmp_order = self.order_set.copy()
                self.tmp_order = np.random.permutation(self.order_set).copy()
            self.cursor = self.tmp_order[:min(self.batch_size, len(self.tmp_order))]
            self.tmp_order = self.tmp_order[min(self.batch_size, len(self.tmp_order)):]

        obs, market_obs, future_return, past_return = self._get_data()
        obs_masks, future_return_masks = self._get_masks(obs, future_return)
        trade_masks = np.logical_or(obs_masks, future_return_masks)
        obs = self._fillna(obs, obs_masks)

        future_ror = np.prod((future_return + 1), axis=-1)
        future_ror[future_return_masks] = 0.
        # future_p = self.__get_p(future_return)
        obs_normed = self.__normalize_assets(obs, obs_masks)
        if self.allow_short:
            market_obs_normed = self.__normalize_market(market_obs)
        else:
            market_obs_normed = None

        self.cursor += self.trade_len
        if self.val_mode:
            done = (self.cursor >= self.test_idx + 1)
        elif self.test_mode:
            done = (self.cursor >= self.__assets_data.shape[1])
        else:
            done = (self.cursor >= self.val_idx).any()
        
        if np.isnan(obs).any():
            print('still have nan not been filled in obs')
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        elif np.isnan(obs_normed).any():
            print('still have nan not been filled in obs_normed')
            obs_normed = np.nan_to_num(obs_normed, nan=0.0, posinf=0.0, neginf=0.0)
        assert not np.isnan(obs + obs_normed).any()

        obs, obs_normed = obs.astype(np.float32), obs_normed.astype(np.float32)
        if self.allow_short:
            market_obs = market_obs.astype(np.float32)
            market_obs_normed = market_obs_normed.astype(np.float32)
        future_ror = future_ror.astype(np.float32)

        return obs, obs_normed, market_obs, market_obs_normed, future_ror, trade_masks, done


    def _get_data(self):
        raw_states = np.zeros(
            (len(self.cursor), self.__assets_data.shape[0], (self.window_len + 1) * 5, self.assets_features))
        assets_states = np.zeros((len(self.cursor), self.__assets_data.shape[0], self.window_len, self.assets_features))
        if self.allow_short:
            market_states = np.zeros((len(self.cursor), self.window_len, self.market_features))
        else:
            market_states = None
        future_return = np.zeros((len(self.cursor), self.__assets_data.shape[0], self.trade_len))
        past_return = np.zeros((len(self.cursor), self.__assets_data.shape[0], self.window_len))
        
        epsilon = 1e-10  # Pequena constante para evitar divisão por zero
        
        for i, idx in enumerate(self.cursor):
            raw_states[i] = self.__assets_data[:, idx - (self.window_len + 1) * 5 + 1:idx + 1].copy()
            tmp_states = raw_states.reshape(raw_states.shape[0], raw_states.shape[1], self.window_len + 1, 5, -1)
            
            # Abaixo, usamos a constante epsilon para evitar divisão por zero
            denominator = tmp_states[i, :, :-1, -1, 0] + epsilon
            assets_states[i, :, :, 0] = tmp_states[i, :, 1:, -1, 0] / denominator
            
            denominator = tmp_states[i, :, 1:, -1, 0] + epsilon
            assets_states[i, :, :, 1] = np.nanmax(tmp_states[i, :, 1:, :, 1], axis=-1) / denominator
            assets_states[i, :, :, 2] = np.nanmin(tmp_states[i, :, 1:, :, 2], axis=-1) / denominator
            assets_states[i, :, :, 3] = np.nansum(tmp_states[i, :, 1:, :, 3], axis=-1)
            assets_states[i, :, :, 4] = np.nanmean(tmp_states[i, :, 1:, :, 4], axis=-1)
            
            if tmp_states.shape[-1] == 6:
                assets_states[i, :, :, 5] = np.nanmean(tmp_states[i, :, 1:, :, 5], axis=-1)
            
            if self.allow_short:
                tmp_states = self.__market_data[idx - (self.window_len) * 5 + 1:idx + 1].reshape(self.window_len, 5, -1)
                tmp_states = tmp_states.astype(float)
                market_states[i] = np.mean(tmp_states, axis=1)
            
            future_return[i] = self.__ror_data[:, idx + 1:min(idx + 1 + self.trade_len, self.__ror_data.shape[-1])]
            past_return[i] = self.__ror_data[:, idx - self.window_len + 1:idx + 1]

            if np.isnan(assets_states).any():
                assets_states = np.nan_to_num(assets_states, nan=0.0, posinf=0.0, neginf=0.0)

        return assets_states, market_states, future_return, past_return


    def _fillna(self, obs, masks):
        """
        :param obs: (batch, num_assets, window_len, features)
        :param masks: bool
        :return:
        """
        obs[masks] = 0.

        in_nan_assets = np.argwhere(np.isnan(np.sum(obs.reshape(obs.shape[0], obs.shape[1], -1), axis=-1)))
        for idx in in_nan_assets:
            tmp_df = pd.DataFrame(obs[idx[0], idx[1]])
            tmp_df = tmp_df.fillna(method='bfill')
            obs[idx[0], idx[1]] = tmp_df.values

        if np.isnan(obs).any():
            print('still have nan not been filled')
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            
        assert not np.isnan(obs).any(), 'still have nan not been filled'
        return obs

    def _get_masks(self, obs_states, future_ror):
        """
        :param obs_states:
        :param future_ror:
        :return:
        """
        obs_masks = np.isnan(obs_states[:, :, -1, 0])
        future_return_masks = np.isnan(np.sum(future_ror, axis=-1))
        return obs_masks, future_return_masks

    def __normalize_assets(self, inputs, masks):
        if self.norm_type == 'standard':
            x_mean = np.mean(inputs, axis=-2, keepdims=True)
            x_std = np.std(inputs, axis=-2, keepdims=True)
            normed = (inputs - x_mean) / (x_std + EPS)
        elif self.norm_type == 'min-max':
            x_max = np.max(inputs, axis=-2, keepdims=True)
            x_min = np.min(inputs, axis=-2, keepdims=True)
            normed = (inputs - x_min) / (x_max - x_min + EPS)
        elif self.norm_type == 'div-last':
            inputs[np.logical_not(masks)] = inputs[np.logical_not(masks)] / inputs[np.logical_not(masks)][:, -1:, :]
            normed = inputs
        else:
            raise NotImplementedError

        return normed

    def __normalize_market(self, inputs):
        """
        :param inputs:
        :return:
        """

        # standarlize
        normed = (inputs - inputs.mean(axis=-2, keepdims=True)) / (inputs.std(axis=-2, keepdims=True))

        return normed

    def __get_p(self, future_return):
        """
        :param future_return:
        :return:
        """
        epsilon = 1e-10  # Pequena constante para evitar divisão por zero
        
        rate_up = np.clip(future_return, 0., math.inf).reshape(future_return.shape[0], -1)
        rate_down = np.clip(future_return, -math.inf, 0.).reshape(future_return.shape[0], -1)
        
        sum_rate_up = np.nansum(rate_up, axis=-1)
        sum_rate_down = np.abs(np.nansum(rate_down, axis=-1))
        
        # Adicionar epsilon ao denominador para evitar divisão por zero
        denominator = sum_rate_up + sum_rate_down + epsilon
        
        # Substituir valores infinitos por um valor grande fixo
        max_value = np.finfo(np.float64).max  # Valor máximo representável
        denominator = np.where(np.isinf(denominator), max_value, denominator)
        
        # Calcular future_p com o denominador ajustado
        future_p = sum_rate_up / denominator
        
        # Substituir possíveis valores NaN resultantes da divisão por zero original
        if np.isnan(future_p).any():
            future_p = np.nan_to_num(future_p, nan=0.0, posinf=0.0, neginf=0.0)
        
        return future_p
    

    def eval(self):
        self.val_mode = True
        self.test_mode = False

    def test(self):
        self.test_mode = True
        self.val_mode = False

    def train(self):
        self.test_mode = False
        self.val_mode = False


class PortfolioSim(object):
    def __init__(self, num_assets, fee, time_cost, allow_short=True):
        self.num_assets = num_assets
        self.fee = fee
        self.time_costs = time_cost
        self.allow_short = allow_short


    def _step(self, w0, ror, p):
        """
        :param w0: (batch, 2 * num_assets)
        :param ror:
        :param p:
        :return:
        """
        assert (p >= 0.0).all() and (p <= 1.0).all()
        dw0 = self.w
        dv0 = self.v
        dcash0 = self.cash
        dstock0 = self.stock
        EPS = 1e-10

        # Substitua valores inválidos por EPS
        ror = np.nan_to_num(ror, nan=EPS, posinf=EPS, neginf=EPS)
        w0 = np.nan_to_num(w0, nan=EPS, posinf=EPS, neginf=EPS)

        if self.allow_short:
            # === short ===
            dv0_short = dv0 * (1 - p)
            dv0_short_after_sale = dv0_short * (1 - self.fee)
            dv0_long = (dv0 * p + dv0_short_after_sale)
            dw0_long = dw0 * ((dv0 * p) / (dv0_long + EPS))[..., None]

            dw0_long_sale = np.clip((dw0_long - w0[:, :self.num_assets]), 0., 1.)
            mu0_long = dw0_long_sale.sum(axis=-1) * self.fee

            sum_ror_w0_long = np.sum(ror * w0[:, :self.num_assets], axis=-1, keepdims=True)
            dw1 = (ror * w0[:, :self.num_assets]) / (sum_ror_w0_long + EPS)

            LongPosition_value = dv0_long * (1 - mu0_long) * (np.sum(ror * w0[:, :self.num_assets], axis=-1) + EPS)
            ShortPosition_value = dv0_short * (np.sum(ror * w0[:, self.num_assets:], axis=-1) + EPS)

            LongPosition_gain = LongPosition_value - dv0_long
            ShortPosition_gain = dv0_short - ShortPosition_value

            LongPosition_return = LongPosition_gain / (dv0_long + EPS)
            ShortPosition_return = ShortPosition_gain / (dv0_short + EPS)

            dv1 = LongPosition_value - (ShortPosition_value / (1 - self.fee + EPS)) + dv0_short

            rate_of_return = dv1 / (dv0 + EPS) - 1
            cash_value = 0.
            stocks_value = dv1

        else:
            dv0 = self.v
            dw0 = self.w

            mu0_long = self.fee * np.sum(np.abs(dw0 - w0[:, :self.num_assets]), axis=-1)

            sum_ror_w0_long = np.sum(ror * w0[:, :self.num_assets], axis=-1, keepdims=True)
            dw1 = (ror * w0[:, :self.num_assets]) / (sum_ror_w0_long + EPS)

            LongPosition_value = dv0 * (1 - mu0_long) * (np.sum(ror * w0[:, :self.num_assets], axis=-1) + EPS)

            LongPosition_gain = LongPosition_value - dv0

            LongPosition_return = LongPosition_gain / (dv0 + EPS)

            dv1 = LongPosition_value

            rate_of_return = dv1 / (dv0 + EPS) - 1

            cash_value = 0.
            stocks_value = LongPosition_value

        r_total = rate_of_return.astype(np.float32)
        r_long = LongPosition_return.astype(np.float32)

        if self.allow_short:
            r_short = ShortPosition_return.astype(np.float32)
        else:
            r_short = None

        reward = Reward(r_total, r_long, r_short)

        self.v = dv1
        self.w = dw1
        self.cash = cash_value
        self.stock = stocks_value

        market_avg_return = np.sum(ror, axis=-1) / (np.sum(ror > 0, axis=-1) + EPS)
        market_avg_return = (market_avg_return - 1).astype(np.float32)

        done = (dv1 == 0).any()
        info = {
            'rate_of_return': rate_of_return,
            'reward': reward,
            'total_value': dv1,
            'market_avg_return': market_avg_return,
            'weights': w0,
            'p': p,
            'market_fluctuation': ror,
        }
        self.inofs.append(info)
        return reward, info, done


    def reset(self, batch_num):
        self.inofs = []
        self.w = np.repeat(np.array([0.] * self.num_assets)[None, ...], repeats=batch_num, axis=0)
        self.v = np.array([1.] * batch_num)
        self.cash = np.array([1.] * batch_num)
        self.stock = np.zeros((batch_num, self.num_assets))


class PortfolioEnv(object):
    def __init__(self,
                 assets_data,
                 market_data,
                 rtns_data,
                 in_features,
                 val_idx,
                 test_idx,
                 batch_size,
                 fee=0.001,
                 time_cost=0.0,
                 window_len=20,
                 trade_len=5,
                 max_steps=20,
                 norm_type='div-last',
                 is_norm=True,
                 allow_short=True,
                 mode='train',
                 assets_name=None,
                 ):

        self.window_len = window_len
        self.num_assets = rtns_data.shape[0]
        self.trade_len = trade_len
        self.val_mode = False
        self.test_mode = False
        self.is_norm = is_norm
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.allow_short = allow_short
        self.mode = mode

        self.src = DataGenerator(assets_data=assets_data, rtns_data=rtns_data, market_data=market_data,
                                 in_features=in_features, val_idx=val_idx, test_idx=test_idx,
                                 batch_size=batch_size, max_steps=max_steps, norm_type=norm_type,
                                 window_len=window_len, trade_len=trade_len, mode=mode, allow_short=allow_short)

        self.sim = PortfolioSim(num_assets=self.num_assets, fee=fee, time_cost=time_cost, allow_short=allow_short)

    def step(self, action, p, simulation=False):
        weights = action
        if simulation:
            raise NotImplementedError
        else:
            obs, obs_normed, market_obs, market_obs_normed, future_ror, future_p, trade_masks, done1 = self.src._step()

        ror = self.ror

        rewards, info, done2 = self.sim._step(weights, ror, p)

        self.ror = future_ror
        if self.is_norm:
            return [obs_normed, market_obs_normed], rewards, future_p, trade_masks, done1 or done2.any(), info
        else:
            return [obs, market_obs], rewards, future_p, trade_masks, done1 or done2.any(), info


    def reset(self):
        self.infos = []
        obs, obs_normed, market_obs, market_obs_normed, future_ror, trade_masks, done = self.src.reset()
        self.sim.reset(obs.shape[0])
        self.ror = future_ror
        # print('obs', obs)
        # print('market_obs', market_obs)

        if self.is_norm:
            return [obs_normed, market_obs_normed], trade_masks
        else:
            return [obs, market_obs], trade_masks


    def set_eval(self):
        self.src.eval()

    def set_test(self):
        self.src.test()

    def set_train(self):
        self.src.train()
