__author__ = 'ssylvain'
# this is the header file where we define the classes
countrylist = ['AUS', 'BRA', 'CAN', 'CHL', 'FRA', 'GER',
                      'ISR', 'ITA', 'JPN', 'MEX', 'ZAF', 'ESP',
                      'SWE', 'GBR', 'USA']


class Asset:  # define super-class to be the Asset of origin
    assetCount = 0

    def __init__(self, country):
        if any(country in s for s in countrylist):
            self.country = country
        else:
            raise NameError('Please use one of the following countries' +
                            ': '.join(countrylist))
        Asset.assetCount += 1
        # print("Calling Asset constructor")

    def getName(self):
        return

    def getTotal(self):
        print("Total # Assets: {}".format(self.assetCount))

    def setCountry(self, country):
        self.country = country

    def getCountry(self):
        return self.country

    def __del__(self):
        class_name = self.__class__.__name__
        # print(class_name, "destroyed")


class Bonds(Asset):  # define child class
    bondCount = 0

    def __init__(self, maturity, country):
        Asset.__init__(self, country)
        if any(country in s for s in countrylist):
            self.country = country
        else:
            raise NameError('Please use one of the following countries' +
                            ': '.join(countrylist))
        self.maturity = maturity
        Bonds.bondCount += 1
        # print("Calling Bonds constructor")
        self.yields = None
        self.yields_dates = None
        self.yields_forecast = None
        self.forecast_se = None
        self.forecast_e = None
        self.forecast_rmse = None

    def setBondMaturity(self, maturity):
        self.maturity = maturity

    def getBondMaturity(self):
        print(self.maturity)

    def printName(self):
        print("Bond with {} years maturity"
              .format(self.maturity))

    def getType(self):
        return 'Bonds'

    def printTotal(self):
        print("Total # Bonds: {}".format(self.bondCount))

    def getTotal(self):
        return self.bondCount

    def setCountry(self, country):
        self.country = country

    def getCountry(self):
        return self.country


class NominalBonds(Bonds):  # define child class
    nomBondCount = 0

    def __init__(self, maturity, country):
        Bonds.__init__(self, maturity, country)
        if any(country in s for s in countrylist):
            self.country = country
        else:
            raise NameError('Please use one of the following countries' +
                            ': '.join(countrylist))
        # super(NominalBonds, self).__init__(self, maturity)
        self.maturity = maturity
        self.nomBondCount += 1
        # self.yields = 0
        # print("Calling NominalBonds constructor")

    def setBondMaturity(self, maturity):
        self.maturity = maturity

    def getBondMaturity(self):
        print(self.maturity)

    def printName(self):
        print("Nominal Bond with {} years maturity"
              .format(self.maturity))

    def getType(self):
        return 'NominalBonds'

    def printTotal(self):
        print("Total Nominal Bonds {}".format(self.nomBondCount))

    def getTotal(self):
        return self.nomBondCount

    def setCountry(self, country):
        self.country = country

    def getCountry(self):
        return self.country

    def setZeroYieldsTS(self, yields):  # time series of zero coupon yields
        self.yields = yields

    def setZeroYieldsDates(self, yields_dates):  # time series of zero coupon yields
        self.yields_dates = yields_dates

    def getZeroYieldsTS(self):  # time series of zero coupon yields
        return self.yields

    def getZeroYieldsDates(self):  # time series of zero coupon yields
        return self.yields_dates


class InfLinkBonds(Bonds):
    infBondCount = 0

    def __init__(self, maturity, country):  # define child class
        Bonds.__init__(self, maturity, country)
        if any(country in s for s in countrylist):
            self.country = country
        else:
            raise NameError('Please use one of the following countries' +
                            ': '.join(countrylist))
        self.maturity = maturity
        self.infBondCount += 1
        # print("Calling InfLinkBonds constructor")

    def setBondMaturity(self, maturity):
        self.maturity = maturity

    def getBondMaturity(self):
        return self.maturity

    def printName(self):
        print("Inf. Linked Bond with {} years maturity"
              .format(self.maturity))

    def getType(self):
        return 'InfLinkBonds'

    def printTotal(self):
        print("Total Inf. Linked Bonds {}".format(self.infBondCount))

    def getTotal(self):
        return self.infBondCount

    def setCountry(self, country):
        self.country = country

    def getCountry(self):
        return self.country

    def setZeroYieldsTS(self, yields):  # time series of zero coupon yields
        self.yields = yields

    def setZeroYieldsDates(self, yields_dates):  # time series of zero coupon yields
        self.yields_dates = yields_dates

    def getZeroYieldsTS(self):  # time series of zero coupon yields
        return self.yields

    def getZeroYieldsDates(self):  # time series of zero coupon yields
        return self.yields_dates