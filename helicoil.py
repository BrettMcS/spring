"""
Design of Springs to EN 13906-1:2013.

Scope. For coil springs where the principal load is axial.

Notation and Units (generally follows the EN notation):

    D   mm    Mean coil diameter
    OD  mm    Outer coil diameter
    d   mm    Bar diameter
    R   N/mm  Axial rate
    F   N     Axial load, in particular the design load (midway defln load)
    G   MPa   Shear Modulus
    E   MPa   Youngs Modulus
    L   mm    Spring length at design load
    L0  mm    Free length (nominal)
    Lc  mm    Solid length
    N   -     Number of active coils
    Nt  -     Number of total coils

"""
import time
from math import pi, sqrt, tan, atan, cos
from scipy.interpolate import interp1d
import numpy as np


# =============================== Spring Geometry ============================

def spring_index(D, d):
    """
    Return the spring index, 'w'.
    """
    return D / d


def total_coils(N, num_inactive_coils=1.5):
    """
    Return the total number of coils, 'Nt'. S9.8
    """
    return N + num_inactive_coils


def active_coils(G, d, D, R):
    """
    Return the number of active coils, 'N'. S9.7 (corrected)
    """
    return G * d**4 / (8.0 * R * D**3)


def Sa_min_reqd_length(D, d, N, dynamic=False):
    """
    Returns Sa the min required spring length beyond solid (static). S9.9
    """
    Sa = 0.02 * N * (D + d)

    if dynamic:
        Sa *= 2.0

    return Sa


def solid_length(N, d_max, inactive_coils=1.5):
    """
    Return the spring solid length, 'Lc'. S9.10
    """
    return (total_coils(N, inactive_coils) - 0.3) * d_max


def diameter_swell(D, L0, Nt):
    """
    Return the increase in outside diameter. S9.11
    """
    circumference = pi * D

    return D * cos(atan(L0 / (circumference * (Nt + 1.75))))  # ?????


def modulus_temp_factor(centigrade):
    """
    Returns change in modulus with given temperature. Fig 4.
    """
    return 1.0 - 0.01 * (centigrade - 20.0) * 7.0 / 250.0


def mass(D, L0, Nt, d, density):
    """
    Return the coil mass in kg.
    """
    rho = density * 1.0e-6  # convert kg/litre to kg/mm^3
    circumference = pi * D
    swell = cos(atan(L0 / (circumference * (Nt + 1.75))))
    bar_length = circumference * Nt / swell
    bar_area = 0.25 * pi * d**2

    return (bar_length - 0.4 * circumference) * bar_area * rho


# =============================== Spring Rates ===============================

def axial_rate(G, d, D, N):
    """
    Return spring axial rate, 'R'. S9.4
    """
    return G * d**4 / (8.0 * N * D**3)


def lateral_rate(G, E, d, D, F, R, L):
    """
    Return spring lateral (shear) rate, 'Sy'
    """
    mod_ratio = G / E

    rig_bend = 0.5 * R * L * D**2 / (1.0 + 2.0 * mod_ratio)
    rig_shear = R * L / mod_ratio

    discriminant = F * (1.0 + F / rig_shear) / rig_bend

    if discriminant > 0.0:
        factor = sqrt(discriminant)
        dist = (1.0 + F / rig_shear) * tan(0.5 * factor * L) / factor
        lat_rate = F / (2.0 * dist - L)
    else:
        lat_rate = 0.0

    return lat_rate


# =============================== Stresses ===================================

def axial_stress_static(D, d, load):
    """
    Return shear stress due to axial compression of spring, for static loads.
    """
    return 8.0 * D * load / (pi * d**3)


def stress_correction(D, d):
    """
    Return the stress correction factor. S
    """
    spg_index = spring_index(D, d)

    return (spg_index + 0.5) / (spg_index - 0.75)


def axial_stress_dynamic(D, d, load):
    """
    Return shear stress due to axial compression of spring, for dynamic loads.
    """
    return stress_correction(D, d) * axial_stress_static(D, d, load)


def lateral_stress(G, E, d, D, F, R, L, lat_defln):
    """
    Return shear stress due to lateral shear of the spring.

    """
    correction = stress_correction(D, d)

    mod_ratio = G / E

    rig_bend = 0.5 * R * L * D**2 / (1.0 + 2.0*mod_ratio)
    rig_shear = R * L / mod_ratio

    discriminant = F * (1.0 + F/rig_shear) / rig_bend

    if discriminant > 0.0:
        factor = sqrt(discriminant)
        dist = (1.0 + F/rig_shear) * tan(0.5 * factor * L) / factor
        lat_rate = F / (2.0*dist - L)
    else:
        lat_rate = 0.0
        dist = 0.0

    return 16.0 * correction * lat_rate * lat_defln * dist / (pi * d**3)


# =============================== Performance Checks =========================

def Buckling_Deflection(G, E, D, freeLength, endCond):
    """
    Return the critical deflection from free L at which buckling occurs.

    :Parameters:
        endCond: float. 2.0:free, to 0.5:fully guided.  See EN figure 5

    """
    mod_ratio = G/E
    Acoeff = 0.5/(1.0 - mod_ratio)
    Bcoeff = (1.0 - mod_ratio)/(0.5 + mod_ratio)
    discriminant = 1.0 - Bcoeff*(pi*D/(endCond*freeLength))**2

    if discriminant < 1.0e-9:
        critDefln = freeLength # that is, no buckling at all
    else:
        critDefln = freeLength*Acoeff*(1.0 - sqrt(discriminant))
        critDefln = min(critDefln, freeLength)

    return critDefln


def fundamental_frequency(N, d, D, G, density):
    """
    Returns frequency [Hz] of fundamental axial vibration mode.

    First convert density to consistent units (t/mm^3) from kg/litre and
    include the 2*pi*sqrt(2) factor.  This results in the 'magic number'
    shown in the EN standard.
    """
    rho = density * 1.0e-9 # convert kg/litre to tonne/mm^3
    factor = 1.0 / (2.0 * pi * sqrt(2.0))

    return factor *d * sqrt(G / rho) / (N * D**2)


# =============================== Material Properties ========================

class GoodmanCurve:
    """
    The EN-style Modified Goodman curve for round bar.
    """

    def __init__(self, barDiam, zeroMinLimit, maxLimit, kneeMin):
        """
        :Parameters:
            barDiam: float. [mm]. Bar diameter
            zeroMinLimit: float. [MPa] Allowable max stress at zero min stress
            maxLimit: float [MPa] Maximum allowable stresses
            kneeMin: float [MPa] Min stress value where upper stress limit
                                 stops increasing.
        """
        self.barDiam = barDiam
        self.zeroMinLimit = zeroMinLimit
        self.maxLimit = maxLimit
        self.kneeMin = kneeMin
        self.slope = (maxLimit - zeroMinLimit) / kneeMin

    def UpperStressLimit(self, minStress):
        """
        Return the allowable maximum stress given a certain min stress.

        """
        if minStress <= 0.0:
            limit = self.zeroMinLimit
        elif minStress >= self.kneeMin:
            limit = self.maxLimit
        else:
            limit = self.zeroMinLimit + self.slope * minStress

        return limit

    def AllowableRange(self, minStress):
        """
        Return the allowable stress range given the minimum stress.

        """
        return self.UpperStressLimit(minStress) - minStress

    def StressRangeReserve(self, minStress, maxStress):
        """
        Return the allowed stress range minus the actual range (can be negative)

        """
        stressRange = maxStress - minStress
        allowableRange = self.AllowableRange(minStress)

        return allowableRange - stressRange


class GoodmanCurves:
    """
    The EN-style Goodman curves for circular bars.
    """
    def __init__(self, barDiams, zeroMinLimits, maxLimits, kneeMins):
        """
        """
        self.barDiamMin = barDiams[0]
        self.barDiamMax = barDiams[-1]

        self.zeroMinLimit = interp1d(barDiams, zeroMinLimits)
        self.maxLimit = interp1d(barDiams, maxLimits)
        self.kneeMin = interp1d(barDiams, kneeMins)

    def GetGoodmanCurve(self, barDiam):
        """
        """
        barDiam = max(barDiam, self.barDiamMin)
        barDiam = min(barDiam, self.barDiamMax)

        return GoodmanCurve(barDiam,
                            self.zeroMinLimit(barDiam),
                            self.maxLimit(barDiam),
                            self.kneeMin(barDiam))


class Material:

    def __init__(self, name, E, G, density,
                 LoCycleGC, HiCycleGC, SolidStressLimit):
        """
        """
        self.name = name
        self.E = E
        self.G = G
        self.density = density
        self.LoCycleGC = LoCycleGC
        self.HiCycleGC = HiCycleGC
        self.SolidStressLimit = SolidStressLimit


prEN10089 = Material(
    name="prEN 10089:2000 special quality steel, ground, shot peened",
    E=206000.0,
    G=78500.0,
    density=7.85,
    LoCycleGC=GoodmanCurves(
        [ 10.0,  15.0,  25.0,  35.0,  50.0],  # bar diameters
        [760.0, 670.0, 590.0, 520.0, 430.0],  # y-intercept
        [890.0, 830.0, 780.0, 740.0, 690.0],  # max stress
        [230.0, 260.0, 300.0, 330.0, 390.0]),  # knee values
    HiCycleGC=GoodmanCurves(
        [ 10.0,  15.0,  25.0,  35.0,  50.0],  # bar diameters
        [645.0, 555.0, 475.0, 405.0, 325.0],  # y-intercept
        [890.0, 830.0, 780.0, 740.0, 690.0],  # max stress
        [390.0, 410.0, 440.0, 460.0, 490.0]),  # knee values
    SolidStressLimit=interp1d(
        [-100.0,   7.5,  10.0,  12.5,  15.0,  20.0,  25.0,  30.0,  40.0,  50.0,  60.6, 200.0],
        [1000.0, 955.0, 925.0, 896.9, 874.4, 840.2, 813.1, 794.6, 761.3, 735.8, 716.5, 700.0])
    )


def FatigueStressReserve(SN_curves, G, d, D, N, L0, L, deflnAmp):
    """
    Return the reserve of actual versus allowed stress range.

    """
    R = axial_rate(G, d, D, N)

    minLoad = R * (L0 - L - deflnAmp)
    maxLoad = R * (L0 - L + deflnAmp)

    minStr = axial_stress_dynamic(D, d, minLoad)
    maxStr = axial_stress_dynamic(D, d, maxLoad)

    GoodmanCurve = SN_curves.GetGoodmanCurve(d)

    fatStrReserve = GoodmanCurve.StressRangeReserve(minStr, maxStr)

    return fatStrReserve


def MinFatigueStressReserve(mat, d, D, N, L0, L, loCycDef, hiCycDef):
    """
    Return the reserve of actual versus allowed stress range.

    """
    loCycRes = FatigueStressReserve(mat.LoCycleGC, mat.G,
                                    d, D, N, L0, L, loCycDef)
    hiCycRes = FatigueStressReserve(mat.HiCycleGC, mat.G,
                                    d, D, N, L0, L, hiCycDef)

    return min(loCycRes, hiCycRes)


def CoilInfoDict(name, d, D, N, F, L, loCycDef, hiCycDef,
                 mat, hotCoiled, groundEnds, closedEnds):
    """
    Returns a dictionary of spring coil information.

    """
    OD = D + d
    ID = D - d
    Nt = total_coils(N, hotCoiled)

    R = axial_rate(mat.G, d, D, N)

    L0 = L + F/R
    Lc = Solid_Length(N, d, hotCoiled, groundEnds, closedEnds)
    Lmin = L - loCycDef
    Lmax = L + loCycDef

    Fmin = F - loCycDef*R
    Fmax = F + loCycDef*R

    Sa = Sa_min_reqd_length(D, d, N, hotCoiled)
    coilGapAtFree = (L0 - Lc)/N

    solidStr = axial_stress_static(D, d, R*(L0-Lc))
    solidStressLimit = mat.SolidStressLimit(d)
    solidStrRes = solidStressLimit - solidStr

    freq = fundamental_frequency(N, d, D, mat.G, mat.density)
    mass = Mass(D, L0, Nt, d, mat.density)

    endCons = np.array([2.0, 1.0, 0.7, 0.5])
    bucklingLens = np.array([L0 - Buckling_Deflection(mat.G, mat.E, D, L0, e)
                            for e in endCons])

    lat_rate = lateral_rate(mat.G, mat.E, d, D, F, R, L)
    lat_rate_minLoad = Lateral_Rate(mat.G, mat.E, d, D, Fmin, R, Lmax)
    lat_rate_maxLoad = Lateral_Rate(mat.G, mat.E, d, D, Fmax, R, Lmin)

    loCycRes = FatigueStressReserve(mat.LoCycleGC, mat.G,
                                    d, D, N, L0, L, loCycDef)
    hiCycRes = FatigueStressReserve(mat.HiCycleGC, mat.G,
                                    d, D, N, L0, L, hiCycDef)

    return dict(name=name,
                time=time.asctime(),
                material=mat.name,
                youngsModulus=mat.E,
                shearModulus=mat.G,
                density=mat.density,
                mass=mass,
                outsideDiameter=OD,
                coilDiameter=D,
                insideDiameter=ID,
                barDiameter=d,
                springIndex=D/d,
                axialRate=R,
                lateralRate_designLoad=lat_rate,
                lateralRate_maxLoad=lat_rate_maxLoad,
                lateralRate_minLoad=lat_rate_minLoad,
                designLoad=F,
                designHeight=L,
                solidHeight=Lc,
                freeHeight=L0,
                actualMinLength=Lmin,
                allowedMinLength=Lc+Sa,
                coilGapAtFree=coilGapAtFree,
                lowCycleDeflection=loCycDef,
                highCycleDeflection=hiCycDef,
                solidStress=solidStr,
                solidStressReserve=solidStrRes,
                lowCycleFatigueReserve=loCycRes,
                highCycleFatigueReserve=hiCycRes,
                activeCoils=N,
                totalCoils=Nt,
                naturalFrequency=freq,
                endSupportCoeffs=endCons,
                bucklingLengths=bucklingLens,
                hotCoiled=hotCoiled,
                groundEnds=groundEnds,
                closedEnds=closedEnds)


def CoilInfoStr(coilInfo):
    """
    Return a string of formatted spring data

    """
    # need to format these arrays first; the default has too much precision
    e = coilInfo['endSupportCoeffs']
    b = coilInfo['bucklingLengths']
    eCons = "[{:5.1f}, {:5.1f}, {:5.1f}, {:5.1f}]".format(e[0],e[1],e[2],e[3])
    bLens = "[{:5.1f}, {:5.1f}, {:5.1f}, {:5.1f}]".format(b[0],b[1],b[2],b[3])

    coilInfo["eCons"] = eCons
    coilInfo["bLens"] = bLens

    designHgt_defln = coilInfo['freeHeight'] - coilInfo['designHeight']

    loCyc_min_defln = designHgt_defln - coilInfo['lowCycleDeflection']
    loCyc_max_defln = designHgt_defln + coilInfo['lowCycleDeflection']

    hiCyc_min_defln = designHgt_defln - coilInfo['highCycleDeflection']
    hiCyc_max_defln = designHgt_defln + coilInfo['highCycleDeflection']

    return """
Coil Name: {name}
Time:      {time}
Notation and Design as per EN 13906-1:2002, Quasi-Static

Material name: {material}
Young's Modulus, E: {youngsModulus:7.0f} MPa
Shear Modulus,   G: {shearModulus:7.0f} MPa
Density,       rho: {density:5.2f} kg/litre

Hot Coiled:  {hotCoiled}
Ground Ends: {groundEnds}
Closed Ends: {closedEnds}

Design Load,   F:{designLoad:8.1f} N
Design Length, L:{designHeight:8.1f} mm

Coil Diameter,     D: {coilDiameter:6.2f}  mm
Bar Diameter,      d: {barDiameter:7.3f} mm
Spring Index,      w: {springIndex:7.3f} -

Outside Diameter, Do: {outsideDiameter:6.2f}  mm
Inside Diameter,  Di: {insideDiameter:6.2f}  mm

Num Active Coils, n: {activeCoils:5.2f}
Num Total Coils, nt: {totalCoils:5.2f}

Axial Rate,    R: {axialRate:5.1f} N/mm
Lateral Rate, RQ: {lateralRate_designLoad:5.1f} N/mm at design load
Lateral Rate    : {lateralRate_maxLoad:5.1f} N/mm at maximum load
Lateral Rate    : {lateralRate_minLoad:5.1f} N/mm at minimum load

Free Length,       L0: {freeHeight:6.1f} mm
Solid Length,      Lc: {solidHeight:6.1f} mm
Min service len,   Lm: {actualMinLength:6.1f} mm
Allowed min len,   Ln: {allowedMinLength:6.1f} mm to maintain coil gap (quasi-static)
Gap between coils, a0: {coilGapAtFree:6.1f} mm at free length, (should be > d)

Seating coeffs,   mu: {eCons}
Buckling lengths, LK: {bLens} mm

The Fatigue deflection amplitudes (about the design height) are:
Lo cycle defln amp: {lowCycleDeflection:5.1f} mm for 1e5 cycles
Hi cycle defln amp: {highCycleDeflection:5.1f} mm for 2e6 cycles

Using the SN Low and High Cycle SN curves for this coil's bar diameter:
Lo cycle fatigue reserve: {lowCycleFatigueReserve:6.1f} MPa below the SN curve
Hi cycle fatigue reserve: {highCycleFatigueReserve:6.1f} MPa below the SN curve

Solid stress:         {solidStress:6.1f} MPa
Solid stress reserve: {solidStressReserve:6.1f} MPa

Natural Frequency, f0: {naturalFrequency:5.1f} Hz
Mass:                  {mass:5.1f} kg
    """.format(**coilInfo)
