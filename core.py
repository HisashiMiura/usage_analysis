from typing import List, Tuple, Optional
import numpy as np
from enum import Enum
from dataclasses import dataclass


class NPerson(Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4

    @property
    def index(self):
        return self.value - 1


class Region(Enum):
    REGION1 = 1
    REGION2 = 2
    REGION3 = 3
    REGION4 = 4
    REGION5 = 5
    REGION6 = 6
    REGION7 = 7
    REGION8 = 8

    @property
    def index(self):
        return self.value - 1


@dataclass
class EP:

    # 月 m における暖房設備の一次エネルギー消費量
    h_m: np.ndarray

    # 月 m における冷房設備の一次エネルギー消費量
    c_m: np.ndarray

    # 月 m における換気設備の一次エネルギー消費量
    v_m: np.ndarray

    # 月 m における照明設備の一次エネルギー消費量
    l_m: np.ndarray

    # 月 m における給湯設備の一次エネルギー消費量
    hw_m: np.ndarray

    # 月 m における家電の一次エネルギー消費量
    ap_m: np.ndarray

    # 月 m における調理の一次エネルギー消費量
    cc_m: np.ndarray

    # 月 m における一次エネルギー消費量の合計
    total_m: np.ndarray


def kernel(
    region: Region, p: NPerson, a_mr: float, a_or: float, a_a: float,
    u_e: bool, u_g: bool, u_k: bool,
    s_e_m: Optional[List[float]] = None,
    u_e_h_m: Optional[List[bool]] = None,
    u_e_c_m: Optional[List[bool]] = None,
    u_e_v: Optional[bool] = None,
    c_e_v: Optional[float] = None,
    u_e_l: Optional[bool] = None,
    c_e_l: Optional[float] = None,
    u_e_hw: Optional[bool] = None,
    c_e_hw: Optional[float] = None,
    u_e_ap: Optional[bool] = None,
    c_e_ap: Optional[float] = None,
    u_e_cc: Optional[bool] = None,
    c_e_cc: Optional[float] = None,
    f_pe_e_default: Optional[bool] = None,
    f_pe_e: Optional[float] = None,
    s_g_m: Optional[List[float]] = None,
    u_g_h_m: Optional[List[bool]] = None,
    u_g_hw: Optional[bool] = None,
    c_g_hw: Optional[float] = None,
    u_g_cc: Optional[bool] = None,
    c_g_cc: Optional[float] = None,
    f_pe_g_default: Optional[bool] = None,
    gas_type: Optional[str] = None,
    f_pe_g: Optional[float] = None,
    s_k_m: Optional[List[float]] = None,
    u_k_h_m: Optional[List[bool]] = None,
    u_k_hw: Optional[bool] = None,
    c_k_hw: Optional[float] = None,
    f_pe_k_default: Optional[bool] = None,
    f_pe_k: Optional[float] = None
) -> Tuple[EP, EP, EP, EP]:

    if u_e:

        # 月 m において暖房設備を使用しているかの有無, bool値, [m], eq.19
        u_h_m = np.array(u_e_h_m)

        # 月 m において冷房設備を使用しているかの有無, bool値, [m], eq.20
        u_c_m = np.array(u_e_c_m)

        # 換気設備の調整係数, eq.21
        c_v = c_e_v if u_e_v else 0.0

        # 照明設備の調整係数, eq.22
        c_l = c_e_l if u_e_l else 0.0

        # 給湯設備の調整係数, eq.23
        c_hw = c_e_hw if u_e_hw else 0.0

        # 家電の調整係数, eq.24
        c_ap = c_e_ap if u_e_ap else 0.0

        # 調理の調整係数, eq.25
        c_cc = c_e_cc if u_e_cc else 0.0

        # 月 m における月別消費量, kWh, [m], eq.26
        s_m = np.array(s_e_m)

        # 月 m における暖房設備の用途の消費量, kWh, [m]
        # 月 m における冷房設備の用途の消費量, kWh, [m]
        # 月 m における換気設備の用途の消費量, kWh, [m]
        # 月 m における照明設備の用途の消費量, kWh, [m]
        # 月 m における給湯設備の用途の消費量, kWh, [m]
        # 月 m における家電の用途の消費量, kWh, [m]
        # 月 m における照明の用途の消費量, kWh, [m]
        s_h_m, s_c_m, s_v_m, s_l_m, s_hw_m, s_ap_m, s_cc_m = core(
            region=region, p=p, a_mr=a_mr, a_or=a_or, a_a=a_a, s_m=s_m, u_h_m=u_h_m, u_c_m=u_c_m, c_v=c_v, c_l=c_l, c_hw=c_hw, c_ap=c_ap, c_cc=c_cc
        )

        # 電気の一次エネルギー換算係数, MJ/kWh
        f_pe_e = 9.76 if f_pe_e_default else f_pe_e

        # 月 m における電気の暖房設備の一次エネルギー消費量, [m], MJ, eq.12
        e_p_e_h_m = s_h_m * f_pe_e

        # 月 m における電気の冷房設備の一次エネルギー消費量, [m], MJ, eq.13
        e_p_e_c_m = s_c_m * f_pe_e

        # 月 m における電気の換気設備の一次エネルギー消費量, [m], MJ, eq.14
        e_p_e_v_m = s_v_m * f_pe_e

        # 月 m における電気の照明設備の一次エネルギー消費量, [m], MJ, eq.15
        e_p_e_l_m = s_l_m * f_pe_e

        # 月 m における電気の給湯設備の一次エネルギー消費量, [m], MJ, eq.16
        e_p_e_hw_m = s_hw_m * f_pe_e

        # 月 m における電気の家電の一次エネルギー消費量, [m], MJ, eq.17
        e_p_e_ap_m = s_ap_m * f_pe_e

        # 月 m における電気の調理の一次エネルギー消費量, [m], MJ, eq.18
        e_p_e_cc_m = s_cc_m * f_pe_e

    else:

        # 月 m における電気の暖房設備の一次エネルギー消費量, [m], MJ
        # 月 m における電気の冷房設備の一次エネルギー消費量, [m], MJ
        # 月 m における電気の換気設備の一次エネルギー消費量, [m], MJ
        # 月 m における電気の照明設備の一次エネルギー消費量, [m], MJ
        # 月 m における電気の給湯設備の一次エネルギー消費量, [m], MJ
        # 月 m における電気の家電の一次エネルギー消費量, [m], MJ
        # 月 m における電気の調理の一次エネルギー消費量, [m], MJ
        e_p_e_h_m, e_p_e_c_m, e_p_e_v_m, e_p_e_l_m, e_p_e_hw_m, e_p_e_ap_m, e_p_e_cc_m =  np.zeros(12), np.zeros(12), np.zeros(12), np.zeros(12), np.zeros(12), np.zeros(12), np.zeros(12)
    
    if u_g:

        # 月 m において暖房設備を使用しているかの有無, bool値, [m], eq.34
        u_h_m = np.array(u_g_h_m)

        # 月 m において冷房設備を使用しているかの有無, bool値, [m], eq.35
        u_c_m=np.full(shape=12, fill_value=False, dtype=bool)

        # 換気設備の調整係数, eq.36
        c_v=0.0

        # 照明設備の調整係数, eq.37
        c_l=0.0
        
        # 給湯設備の調整係数, eq.38
        c_hw = c_g_hw if u_g_hw else 0.0

        # 家電の調整係数, eq.39
        c_ap=0.0

        # 調理の調整係数, eq.40
        c_cc = c_g_cc if u_g_cc else 0.0

        # 月 m における月別消費量, m3, eq.41
        s_m = np.array(s_g_m)

        # 月 m における暖房設備の用途の消費量, m3, [m]
        # 月 m における冷房設備の用途の消費量, m3, [m]
        # 月 m における換気設備の用途の消費量, m3, [m]
        # 月 m における照明設備の用途の消費量, m3, [m]
        # 月 m における給湯設備の用途の消費量, m3, [m]
        # 月 m における家電の用途の消費量, m3, [m]
        # 月 m における照明の用途の消費量, m3, [m]
        s_h_m, s_c_m, s_v_m, s_l_m, s_hw_m, s_ap_m, s_cc_m = core(
            region=region, p=p, a_mr=a_mr, a_or=a_or, a_a=a_a, s_m=s_m, u_h_m=u_h_m, u_c_m=u_c_m, c_v=c_v, c_l=c_l, c_hw=c_hw, c_ap=c_ap, c_cc=c_cc
        )

        # ガスの一次エネルギー換算係数, MJ/m3
        f_pe_g = {'city_gas': 45.0, 'LP': 100.0}[gas_type] if f_pe_g_default else f_pe_g

        # 月 m におけるガスの暖房設備の一次エネルギー消費量, [m], MJ, eq.27
        e_p_g_h_m = s_h_m * f_pe_g

        # 月 m におけるガスの冷房設備の一次エネルギー消費量, [m], MJ, eq.28
        e_p_g_c_m = s_c_m * f_pe_g

        # 月 m におけるガスの換気設備の一次エネルギー消費量, [m], MJ, eq.29
        e_p_g_v_m = s_v_m * f_pe_g

        # 月 m におけるガスの照明設備の一次エネルギー消費量, [m], MJ, eq.30
        e_p_g_l_m = s_l_m * f_pe_g

        # 月 m におけるガスの給湯設備の一次エネルギー消費量, [m], MJ, eq.31
        e_p_g_hw_m = s_hw_m * f_pe_g

        # 月 m におけるガスの家電の一次エネルギー消費量, [m], MJ, eq.32
        e_p_g_ap_m = s_ap_m * f_pe_g

        # 月 m におけるガスの調理の一次エネルギー消費量, [m], MJ, eq.33
        e_p_g_cc_m = s_cc_m * f_pe_g

    else:

        # 月 m におけるガスの暖房設備の一次エネルギー消費量, [m], MJ
        # 月 m におけるガスの冷房設備の一次エネルギー消費量, [m], MJ
        # 月 m におけるガスの換気設備の一次エネルギー消費量, [m], MJ
        # 月 m におけるガスの照明設備の一次エネルギー消費量, [m], MJ
        # 月 m におけるガスの給湯設備の一次エネルギー消費量, [m], MJ
        # 月 m におけるガスの家電の一次エネルギー消費量, [m], MJ
        # 月 m におけるガスの調理の一次エネルギー消費量, [m], MJ
        e_p_g_h_m, e_p_g_c_m, e_p_g_v_m, e_p_g_l_m, e_p_g_hw_m, e_p_g_ap_m, e_p_g_cc_m =  np.zeros(12), np.zeros(12), np.zeros(12), np.zeros(12), np.zeros(12), np.zeros(12), np.zeros(12)
    
    if u_k:

        # 月 m において暖房設備を使用しているかの有無, bool値, [m], eq.49
        u_h_m = np.array(u_k_h_m)

        # 月 m において冷房設備を使用しているかの有無, bool値, [m], eq.50
        u_c_m=np.full(shape=12, fill_value=False, dtype=bool)

        # 換気設備の調整係数, eq.51
        c_v=0.0

        # 照明設備の調整係数, eq.52
        c_l=0.0

        # 給湯設備の調整係数, eq.53
        c_hw = c_k_hw if u_k_hw else 0.0

        # 家電の調整係数, eq.54
        c_ap=0.0

        # 調理の調整係数, eq.55
        c_cc=0.0        
        
        # 月 m における月別消費量, L, eq.56
        s_m = np.array(s_k_m)
  
        # 月 m における暖房設備の用途の消費量, L, [m]
        # 月 m における冷房設備の用途の消費量, L, [m]
        # 月 m における換気設備の用途の消費量, L, [m]
        # 月 m における照明設備の用途の消費量, L, [m]
        # 月 m における給湯設備の用途の消費量, L, [m]
        # 月 m における家電の用途の消費量, L, [m]
        # 月 m における照明の用途の消費量, L, [m]
        s_h_m, s_c_m, s_v_m, s_l_m, s_hw_m, s_ap_m, s_cc_m = core(
            region=region, p=p, a_mr=a_mr, a_or=a_or, a_a=a_a, s_m=s_m, u_h_m=u_h_m, u_c_m=u_c_m, c_v=c_v, c_l=c_l, c_hw=c_hw, c_ap=c_ap, c_cc=c_cc
        )

        # 灯油の一次エネルギー換算係数, MJ/L
        f_pe_k = 37.0 if f_pe_k_default else f_pe_k

        # 月 m における灯油の暖房設備の一次エネルギー消費量, [m], MJ, eq.42
        e_p_k_h_m = s_h_m * f_pe_k

        # 月 m における灯油の冷房設備の一次エネルギー消費量, [m], MJ, eq.43
        e_p_k_c_m = s_c_m * f_pe_k

        # 月 m における灯油の換気設備の一次エネルギー消費量, [m], MJ, eq.44
        e_p_k_v_m = s_v_m * f_pe_k

        # 月 m における灯油の照明設備の一次エネルギー消費量, [m], MJ, eq.45
        e_p_k_l_m = s_l_m * f_pe_k

        # 月 m における灯油の給湯設備の一次エネルギー消費量, [m], MJ, eq.46
        e_p_k_hw_m = s_hw_m * f_pe_k

        # 月 m における灯油の家電の一次エネルギー消費量, [m], MJ, eq.47
        e_p_k_ap_m = s_ap_m * f_pe_k

        # 月 m における灯油の調理の一次エネルギー消費量, [m], MJ, eq.48 
        e_p_k_cc_m = s_cc_m * f_pe_k
 
    else:

        # 月 m における灯油の暖房設備の一次エネルギー消費量, [m], MJ
        # 月 m における灯油の冷房設備の一次エネルギー消費量, [m], MJ
        # 月 m における灯油の換気設備の一次エネルギー消費量, [m], MJ
        # 月 m における灯油の照明設備の一次エネルギー消費量, [m], MJ
        # 月 m における灯油の給湯設備の一次エネルギー消費量, [m], MJ
        # 月 m における灯油の家電の一次エネルギー消費量, [m], MJ
        # 月 m における灯油の調理の一次エネルギー消費量, [m], MJ
        e_p_k_h_m, e_p_k_c_m, e_p_k_v_m, e_p_k_l_m, e_p_k_hw_m, e_p_k_ap_m, e_p_k_cc_m =  np.zeros(12), np.zeros(12), np.zeros(12), np.zeros(12), np.zeros(12), np.zeros(12), np.zeros(12)

    # 月 m における暖房設備の一次エネルギー消費量, [m], MJ, eq.5
    e_p_h_m = e_p_e_h_m + e_p_g_h_m + e_p_k_h_m

    # 月 m における冷房設備の一次エネルギー消費量, [m], MJ, eq.6
    e_p_c_m = e_p_e_c_m + e_p_g_c_m + e_p_k_c_m

    # 月 m における換気設備の一次エネルギー消費量, [m], MJ, eq.7
    e_p_v_m = e_p_e_v_m + e_p_g_v_m + e_p_k_v_m

    # 月 m における照明設備の一次エネルギー消費量, [m], MJ, eq.8
    e_p_l_m = e_p_e_l_m + e_p_g_l_m + e_p_k_l_m

    # 月 m における給湯設備の一次エネルギー消費量, [m], MJ, eq.9
    e_p_hw_m = e_p_e_hw_m + e_p_g_hw_m + e_p_k_hw_m

    # 月 m における家電の一次エネルギー消費量, [m], MJ, eq.10
    e_p_ap_m = e_p_e_ap_m + e_p_g_ap_m + e_p_k_ap_m

    # 月 m における調理の一次エネルギー消費量, [m], MJ, eq.11
    e_p_cc_m = e_p_e_cc_m + e_p_g_cc_m + e_p_k_cc_m

    # 月 m における電気の一次エネルギー消費量（合計）, [m] , MJ, eq.2
    e_p_e_total_m = e_p_e_h_m + e_p_e_c_m + e_p_e_v_m + e_p_e_l_m + e_p_e_hw_m + e_p_e_ap_m + e_p_e_cc_m

    # 月 m におけるガスの一次エネルギー消費量（合計）, [m] , MJ, eq.3
    e_p_g_total_m = e_p_g_h_m + e_p_g_c_m + e_p_g_v_m + e_p_g_l_m + e_p_g_hw_m + e_p_g_ap_m + e_p_g_cc_m

    # 月 m における灯油の一次エネルギー消費量（合計）, [m] , MJ, eq.4
    e_p_k_total_m = e_p_k_h_m + e_p_k_c_m + e_p_k_v_m + e_p_k_l_m + e_p_k_hw_m + e_p_k_ap_m + e_p_k_cc_m

    # 月 m における一次エネルギー消費量（合計）, [m] , MJ, eq.1
    e_p_total_m = e_p_h_m + e_p_c_m + e_p_v_m + e_p_l_m + e_p_hw_m + e_p_ap_m + e_p_cc_m

    e_p_e = EP(
        h_m=e_p_e_h_m,
        c_m=e_p_e_c_m,
        v_m=e_p_e_v_m,
        l_m=e_p_e_l_m,
        hw_m=e_p_e_hw_m,
        ap_m=e_p_e_ap_m,
        cc_m=e_p_e_cc_m,
        total_m=e_p_e_total_m
    )

    e_p_g = EP(
        h_m=e_p_g_h_m,
        c_m=e_p_g_c_m,
        v_m=e_p_g_v_m,
        l_m=e_p_g_l_m,
        hw_m=e_p_g_hw_m,
        ap_m=e_p_g_ap_m,
        cc_m=e_p_g_cc_m,
        total_m=e_p_g_total_m
    )

    e_p_k = EP(
        h_m=e_p_k_h_m,
        c_m=e_p_k_c_m,
        v_m=e_p_k_v_m,
        l_m=e_p_k_l_m,
        hw_m=e_p_k_hw_m,
        ap_m=e_p_k_ap_m,
        cc_m=e_p_k_cc_m,
        total_m=e_p_k_total_m
    )

    e_p = EP(
        h_m=e_p_h_m,
        c_m=e_p_c_m,
        v_m=e_p_v_m,
        l_m=e_p_l_m,
        hw_m=e_p_hw_m,
        ap_m=e_p_ap_m,
        cc_m=e_p_cc_m,
        total_m=e_p_total_m
    )

    return e_p_e, e_p_g, e_p_k, e_p


def core(
    region: Region, p: NPerson, a_mr: float, a_or: float, a_a: float, s_m: np.ndarray,
    u_h_m: np.ndarray, u_c_m: np.ndarray, c_v: float, c_l: float, c_hw: float, c_ap: float, c_cc: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """月別の参照一次エネルギー消費量を計算する。

    Args:
        region: 地域の区分
        p: 居住人数
        a_mr: 主たる居室の床面積, m2
        a_or: その他の居室の床面積, m2
        a_a: 床面積の合計, m2
        s_m: 月 m における月別消費量, [m]
        u_h_m: 月 m において暖房設備を使用しているかの有無, [m]
        u_c_m: 月 m において冷房設備を使用しているかの有無, [m]
        c_v: 換気設備の調整係数
        c_l: 照明設備の調整係数
        c_hw: 給湯設備の調整係数
        c_ap: 家電の調整係数
        c_cc: 調理の調整係数

    Returns:
        月 m における暖房設備の用途の消費量, [m]
        月 m における冷房設備の用途の消費量, [m]
        月 m における換気設備の用途の消費量, [m]
        月 m における照明設備の用途の消費量, [m]
        月 m における給湯設備の用途の消費量, [m]
        月 m における家電の用途の消費量, [m]
        月 m における調理の用途の消費量, [m]
    """

    check_value(heating_use=u_h_m, cooling_use=u_c_m)
    
    # 月 m が中間月か否か, [m]
    im_m = get_im_m(u_h_m=u_h_m, u_c_m=u_c_m)

    # 月 m における換気設備の参照一次エネルギー消費量, [m], MJ
    e_v_ref_m = get_e_v_ref_m(a_a=a_a, p=p, c_v=c_v)

    # 月 m における照明設備の参照一次エネルギー消費量, [m], MJ
    e_l_ref_m = get_e_l_ref_m(a_a=a_a, a_mr=a_mr, a_or=a_or, p=p, c_l=c_l)

    # 月 m における給湯設備の参照一次エネルギー消費量, [m], MJ
    e_hw_ref_m = get_e_hw_ref_m(region=region, p=p, c_hw=c_hw)

    # 月 m における家電の参照一次エネルギー消費量, [m], MJ
    e_ap_ref_m = get_e_ap_ref_m(p=p, c_ap=c_ap)

    # 月 m における調理の参照一次エネルギー消費量, [m], MJ
    e_cc_ref_m = get_e_cc_ref_m(p=p, c_cc=c_cc)
    
    # 月 m における参照一次エネルギー消費量, [m], MJ, eq.68
    e_exhc_ref_m = e_v_ref_m + e_l_ref_m + e_hw_ref_m + e_ap_ref_m + e_cc_ref_m

    # 月 m における暖冷房以外の用途の参照消費量が暖冷房以外の用途の参照消費量の年間合計値に占める割合, [m], eq.67
    if e_exhc_ref_m.sum() == 0.0:
        r_s_exhc_ref_m = np.zeros(shape=12, dtype=float)
    else:
        r_s_exhc_ref_m = e_exhc_ref_m / e_exhc_ref_m.sum()

    def div(x, y): return 0.0 if y == 0.0 else x / y
        
    # 月 m における換気設備の消費量が月mにおける暖冷房以外の用途の消費量に占める割合, [m], eq.66a
    r_s_v_m = np.vectorize(div)(e_v_ref_m, e_exhc_ref_m)

    # 月 m における照明設備の消費量が月mにおける暖冷房以外の用途の消費量に占める割合, [m], eq.66b
    r_s_l_m = np.vectorize(div)(e_l_ref_m, e_exhc_ref_m)

    # 月 m における給湯設備の消費量が月mにおける暖冷房以外の用途の消費量に占める割合, [m], eq.66c
    r_s_hw_m = np.vectorize(div)(e_hw_ref_m, e_exhc_ref_m)

    # 月 m における家電の消費量が月mにおける暖冷房以外の用途の消費量に占める割合, [m], eq.66d
    r_s_ap_m = np.vectorize(div)(e_ap_ref_m, e_exhc_ref_m)

    # 月 m における調理の消費量が月mにおける暖冷房以外の用途の消費量に占める割合, [m], eq.66e
    r_s_cc_m = np.vectorize(div)(e_cc_ref_m, e_exhc_ref_m)

    # 暖冷房設備以外の用途の消費量
    if True in im_m:
        # eq.65a
        s_exhc = div(np.sum(s_m[im_m == True]), np.sum(r_s_exhc_ref_m[im_m == True]))
    else:
        # eq.65b
        s_exhc = div(s_m[s_m.argmin()], r_s_exhc_ref_m[s_m.argmin()])

    # 月 m における暖冷房設備以外の用途の消費量, [m], eq.64    
    s_exhc_m = np.empty(shape=12, dtype=float)
    s_exhc_m[im_m] = s_m[im_m]
    s_exhc_m[im_m == False] = np.minimum(s_m, s_exhc * r_s_exhc_ref_m)[im_m == False]

    # 月 m における暖房設備の用途の消費量, [m], eq.57
    s_h_m = np.zeros(shape=12, dtype=float)
    s_h_m[u_h_m] = (s_m - s_exhc_m)[u_h_m]

    # 月 m における冷房設備の用途の消費量, [m], eq.58
    s_c_m = np.zeros(shape=12, dtype=float)
    s_c_m[u_c_m] = (s_m - s_exhc_m)[u_c_m]

    # 月 m における換気設備の用途の消費量, [m], eq.59
    s_v_m = r_s_v_m * s_exhc_m

    # 月 m における照明設備の用途の消費量, [m], eq.60
    s_l_m = r_s_l_m * s_exhc_m

    # 月 m における給湯設備の用途の消費量, [m], eq.61
    s_hw_m = r_s_hw_m * s_exhc_m

    # 月 m における家電の用途の消費量, [m], eq.62
    s_ap_m = r_s_ap_m * s_exhc_m

    # 月 m における調理の用途の消費量, [m], eq.63
    s_cc_m = r_s_cc_m * s_exhc_m

    return s_h_m, s_c_m, s_v_m, s_l_m, s_hw_m, s_ap_m, s_cc_m


def check_value(heating_use, cooling_use):

    if len(heating_use) != 12:
        raise ValueError('ERROR: 暖房の使用の有無が12ヶ月分ありません。')
    
    if len(cooling_use) != 12:
        raise ValueError('ERROR: 冷房の使用の有無が12ヶ月分ありません。')
    
    for h,c in zip(heating_use, cooling_use):
        if h and c:
            raise ValueError('エラー：同じ月に暖房使用月と冷房使用月は同時に設定できません。(' + str(i) + '月)' ) 
    

def get_all_use(htg_use, ckg_use, consumption, v_l_e_c):
    
    htg = [c-n if u else 0.0 for (c,n,u) in zip(consumption, v_l_e_c, htg_use)]
    clg = [c-n if u else 0.0 for (c,n,u) in zip(consumption, v_l_e_c, ckg_use)]
    
    return htg, clg


def get_e_v_ref_m(a_a: float, p: NPerson, c_v: float) -> np.ndarray:
    """月別の換気設備の参照一次エネルギー消費量を計算する。

    Args:
        a_a: 床面積の合計, m2
        p: 居住人数
        c_v: 換気設備の調整係数

    Returns:
        月 m における換気設備の参照一次エネルギー消費量, MJ
    """

    # table 2a
    alpha_v_m = np.array([
        2.87553024, 2.59725312, 2.87553024,
        2.78277120, 2.87553024, 2.78277120,
        2.87553024, 2.87553024, 2.78277120,
        2.87553024, 2.78277120, 2.87553024
    ])

    # table 2b
    beta_v_p_m = np.array([
        [11.0890192, 9.97432960, 11.0890192, 10.5689088, 10.7698672, 10.6953984, 11.1551920, 11.0559328, 10.5689088, 10.9294432, 10.7615712, 10.9294432],
        [22.1551024, 19.9277728, 22.1551024, 21.1146864, 21.5150416, 21.3685440, 22.2874480, 22.0889296, 21.1146864, 21.8350720, 21.5008896, 21.8350720],
        [33.2506608, 29.9079584, 33.2506608, 31.6898416, 32.2912528, 32.0701888, 33.4493744, 33.1513040, 31.6898416, 32.7709568, 32.2689024, 32.7709568],
        [44.3076672, 39.8532032, 44.3076672, 42.2268352, 43.0273504, 42.7345504, 44.5725536, 44.1752240, 42.2268352, 43.6675088, 42.9994368, 43.6675088]
    ])

    # eq.69
    return (alpha_v_m * a_a + beta_v_p_m[p.index]) * c_v


def get_e_l_ref_m(a_a: float, a_mr: float, a_or: float, p: NPerson, c_l: float) -> np.ndarray:
    """月別の照明設備の参照一次エネルギー消費量を計算する。

    Args:
        a_a: 床面積の合計, m2
        a_mr: 主たる居室の床面積, m2
        a_or: その他の居室の床面積, m2
        p: 居住人数
        c_l: 照明設備の調整係数

    Returns:
        月 m における照明設備の参照一次エネルギー消費量, MJ
    """

    # table 3a
    alpha_p_l_m = np.array([
        [2.394, 2.124, 2.394, 2.353, 2.409, 2.223, 2.639, 2.272, 2.353, 2.402, 2.467, 2.402],
        [4.965, 4.458, 4.965, 4.729, 4.807, 4.773, 5.036, 4.93, 4.729, 4.886, 4.843, 4.886],
        [5.634, 5.062, 5.634, 5.352, 5.436, 5.428, 5.681, 5.61, 5.352, 5.535, 5.475, 5.535],
        [6.303, 5.666, 6.303, 5.976, 6.066, 6.083, 6.326, 6.291, 5.976, 6.184, 6.106, 6.184]
    ])

    # table 3b
    beta_p_l_m = np.array([
        [0.139, 0.125, 0.139, 0.137, 0.143, 0.133, 0.143, 0.137, 0.137, 0.141, 0.137, 0.141],
        [0.286, 0.258, 0.286, 0.274, 0.282, 0.278, 0.282, 0.288, 0.274, 0.284, 0.274, 0.284],
        [1.345, 1.202, 1.345, 1.294, 1.317, 1.276, 1.409, 1.313, 1.294, 1.331, 1.34, 1.331],
        [2.405, 2.146, 2.405, 2.314, 2.352, 2.274, 2.537, 2.339, 2.314, 2.378, 2.406, 2.378]
    ])

    # table 3c
    gamma_p_l_m = np.array([
        [0.272, 0.245, 0.272, 0.271, 0.284, 0.259, 0.284, 0.266, 0.271, 0.278, 0.271, 0.278],
        [0.661, 0.598, 0.661, 0.636, 0.655, 0.642, 0.653, 0.665, 0.636, 0.658, 0.635, 0.658],
        [0.826, 0.744, 0.826, 0.794, 0.814, 0.797, 0.83, 0.824, 0.794, 0.82, 0.801, 0.82],
        [0.991, 0.891, 0.991, 0.951, 0.974, 0.952, 1.007, 0.983, 0.951, 0.982, 0.968, 0.982]
    ])

    # table 3d
    delta_p_l_m = np.array([
        [54.52, 48.65, 54.52, 53.82, 55.58, 51.11, 58.89, 52.34, 53.82, 55.05, 55.47, 55.05],
        [121.51, 109.4, 121.51, 116.08, 118.58, 117.32, 121.98, 121.28, 116.08, 120.04, 117.78, 120.04],
        [172.18, 154.65, 172.18, 164.4, 167.46, 165.48, 174.75, 170.9, 164.4, 169.82, 168.05, 169.82],
        [222.85, 199.9, 222.85, 212.72, 216.34, 213.65, 227.51, 220.52, 212.72, 219.59, 218.31, 219.59]
    ])

    idx_p = p.index

    # eq.71
    a_nr = a_a - a_mr - a_or

    # eq.70
    return (alpha_p_l_m[idx_p] * a_mr + beta_p_l_m[idx_p] * a_or + gamma_p_l_m[idx_p] * a_nr + delta_p_l_m[idx_p]) * c_l


def get_e_hw_ref_m(region: Region, p: NPerson, c_hw: float) -> np.ndarray:
    """月別の給湯設備の参照一次エネルギー消費量を計算する。

    Args:
        region: 地域の区分
        p: 居住人数
        c_hw: 給湯設備の調整係数

    Returns:
        月 m における給湯設備の参照一次エネルギー消費量, MJ
    """

    table4a = [
        [1277.1, 1261.8, 1172.1, 1138.0, 1213.6, 1129.8, 1024.0, 699.5],
        [1139.1, 1130.5, 1051.0, 1034.3, 1096.0, 1006.9,  901.6, 636.5],
        [1251.8, 1247.1, 1115.4, 1110.2, 1154.8, 1080.2,  958.5, 696.8],
        [1027.9, 1004.4,  940.5,  915.0,  908.7,  829.8,  732.3, 585.7],
        [ 924.3,  908.2,  834.7,  799.6,  755.0,  709.6,  638.5, 522.3],
        [ 819.9,  826.0,  759.6,  726.0,  681.3,  582.4,  545.3, 487.1],
        [ 815.6,  787.3,  752.8,  724.9,  639.5,  524.3,  465.6, 459.7],
        [ 706.7,  687.9,  662.6,  624.4,  549.2,  449.6,  423.5, 431.7],
        [ 778.9,  739.5,  699.4,  680.9,  591.8,  532.7,  459.8, 435.9],
        [ 943.5,  906.5,  861.3,  827.6,  789.7,  700.8,  636.1, 496.5],
        [1064.5, 1040.6,  979.9,  958.3,  954.1,  871.8,  814.7, 585.5],
        [1199.9, 1162.5, 1073.8, 1047.6, 1135.7, 1008.1,  919.0, 644.1]
    ]

    table4b = [
        [2049.2, 2014.5, 1866.8, 1811.0, 1902.2, 1772.9, 1610.9, 1106.7],
        [1826.3, 1803.2, 1674.0, 1646.9, 1719.1, 1581.8, 1421.5, 1008.0],
        [1999.1, 1987.7, 1775.7, 1767.0, 1812.5, 1697.6, 1510.2, 1104.4],
        [1628.5, 1592.6, 1490.4, 1445.7, 1429.0, 1305.3, 1155.0,  925.7],
        [1460.5, 1435.8, 1318.4, 1260.1, 1188.1, 1116.5, 1008.6,  826.3],
        [1287.9, 1296.3, 1188.1, 1132.9, 1069.5,  914.2,  859.0,  767.0],
        [1281.9, 1231.8, 1175.1, 1131.1, 1004.5,  825.6,  738.8,  726.9],
        [1103.0, 1070.7, 1027.1,  966.6,  863.3,  710.2,  671.7,  682.0],
        [1230.4, 1167.7, 1098.9, 1070.1,  934.4,  841.2,  731.6,  690.1],
        [1496.4, 1435.4, 1364.3, 1309.3, 1246.6, 1108.3, 1009.7,  786.5],
        [1697.9, 1657.8, 1560.6, 1524.7, 1505.3, 1376.8, 1288.5,  928.0],
        [1918.4, 1854.2, 1714.0, 1669.6, 1785.7, 1591.1, 1451.4, 1021.9]
    ]

    table4c = [
        [3031.3, 2978.6, 2764.8, 2685.6, 2853.4, 2660.1, 2417.2, 1668.1],
        [2709.7, 2674.9, 2486.5, 2448.4, 2584.8, 2378.9, 2138.7, 1522.5],
        [2932.9, 2916.4, 2610.9, 2596.2, 2696.1, 2526.9, 2248.7, 1648.1],
        [2425.6, 2374.1, 2224.1, 2158.0, 2153.6, 1969.0, 1747.0, 1400.8],
        [2194.4, 2157.9, 1985.8, 1897.6, 1806.8, 1700.1, 1538.1, 1258.8],
        [1932.9, 1942.3, 1783.0, 1700.7, 1620.1, 1386.2, 1303.8, 1159.7],
        [1880.4, 1810.4, 1728.0, 1660.6, 1488.6, 1222.9, 1093.0, 1076.6],
        [1655.4, 1608.9, 1540.9, 1449.2, 1305.6, 1073.8, 1018.4, 1031.0],
        [1838.5, 1746.6, 1645.1, 1600.3, 1411.7, 1273.7, 1108.0, 1041.1],
        [2232.5, 2142.2, 2037.0, 1957.4, 1883.6, 1678.5, 1529.1, 1189.9],
        [2512.4, 2452.3, 2311.4, 2257.5, 2253.3, 2061.5, 1930.1, 1395.2],
        [2855.1, 2762.2, 2556.5, 2490.1, 2691.7, 2403.1, 2196.1, 1550.2]
    ]

    table4d = [
        [3377.2, 3319.7, 3084.6, 2997.5, 3211.2, 2994.4, 2720.5, 1877.3],
        [3021.8, 2983.9, 2776.0, 2734.1, 2910.5, 2679.2, 2408.5, 1714.4],
        [3267.5, 3249.4, 2912.6, 2896.0, 3031.3, 2841.1, 2528.0, 1852.7],
        [2710.3, 2653.7, 2486.9, 2412.8, 2425.0, 2217.1, 1967.8, 1577.1],
        [2456.3, 2415.6, 2224.0, 2125.0, 2037.1, 1916.9, 1734.9, 1421.2],
        [2163.7, 2173.8, 1996.0, 1904.1, 1825.2, 1561.0, 1469.3, 1309.9],
        [2097.3, 2018.7, 1926.5, 1853.3, 1671.2, 1376.9, 1231.4, 1213.4],
        [1854.1, 1801.5, 1727.2, 1626.4, 1473.9, 1215.4, 1153.1, 1167.3],
        [2056.9, 1954.3, 1841.0, 1791.4, 1589.9, 1434.7, 1250.3, 1177.8],
        [2496.7, 2395.7, 2278.5, 2190.0, 2122.0, 1891.2, 1723.0, 1343.4],
        [2803.2, 2736.7, 2580.8, 2520.8, 2533.8, 2317.5, 2169.6, 1568.2],
        [3187.1, 3084.8, 2857.6, 2784.2, 3033.1, 2708.8, 2475.9, 1747.1]
    ]

    ref_table = {
        NPerson.ONE: table4a,
        NPerson.TWO: table4b,
        NPerson.THREE: table4c,
        NPerson.FOUR: table4d
    }[p]

    return ((np.array(ref_table).T)[region.index]) * c_hw


def get_e_ap_ref_m(p: NPerson, c_ap: float) -> np.ndarray:
    """月別の家電の参照一次エネルギー消費量を計算する。

    Args:
        p: 居住人数
        c_ap: 家電の調整係数

    Returns:
        月 m における家電の参照一次エネルギー消費量, MJ
    """

    table5 = [
        [849.7, 1031.2, 1409.4, 1481.9],
        [765.3,  928.7, 1268.9, 1333.7],
        [849.7, 1031.2, 1409.4, 1481.9],
        [815.6,  988.7, 1352.0, 1424.0],
        [836.1, 1012.5, 1384.8, 1459.5],
        [819.9,  995.1, 1359.1, 1427.1],
        [854.6, 1037.2, 1419.7, 1498.0],
        [847.2, 1028.3, 1404.2, 1473.8],
        [815.6,  988.7, 1352.0, 1424.0],
        [842.9, 1021.9, 1397.1, 1470.7],
        [824.9, 1001.0, 1369.5, 1443.2],
        [842.9, 1021.9, 1397.1, 1470.7]
    ]

    return ((np.array(table5).T)[p.index]) * c_ap


def get_e_cc_ref_m(p:NPerson, c_cc: float) -> np.ndarray:
    """月別の調理の参照一次エネルギー消費量を計算する。

    Args:
        p: 居住人数
        c_cc: 調理の調整係数

    Returns:
        月 m における調理の参照一次エネルギー消費量, MJ
    """

    table6 = [
        [191.0, 235.3, 284.0, 332.7],
        [171.1, 211.3, 255.1, 298.8],
        [191.0, 235.3, 284.0, 332.7],
        [179.5, 220.1, 265.7, 311.2],
        [180.6, 221.2, 267.0, 312.8],
        [183.6, 227.6, 274.7, 321.9],
        [193.1, 234.3, 282.8, 331.4],
        [189.9, 235.7, 284.6, 333.4],
        [179.5, 220.1, 265.7, 311.2],
        [185.8, 228.2, 275.5, 322.8],
        [185.8, 226.6, 273.6, 320.5],
        [185.8, 228.2, 275.5, 322.8]
    ]

    return ((np.array(table6).T)[p.index]) * c_cc


def get_im_m(u_h_m: np.ndarray, u_c_m: np.ndarray) -> np.ndarray:
    """中間月を取得する。

    Args:
        u_h_ms: 月 m において暖房を使用しているかの有無, [m]
        u_c_ms: 月 m において冷房を使用しているかの有無, [m]

    Returns:
        月 m が中間月か否か, [m]
    
    Notes:
        m は 12である。
    """

    # eq.72    
    return (u_h_m == False) & (u_c_m == False)
