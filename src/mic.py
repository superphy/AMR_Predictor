#!/usr/bin/env python

"""mics.py

Classes to parse MIC data

"""

import re
import numpy as np
import warnings
from collections import Counter

__author__ = "Matthew Whiteside"
__copyright__ = "Copyright 2018, Public Health Agency of Canada"
__license__ = "APL"
__version__ = "2.0"
__maintainer__ = "Matthew Whiteside"
__email__ = "matthew.whiteside@phac-aspc.gc.ca"


class MICPanel:
    """MIC Panel Class

    Class to handle MIC inputs for a single drug

    """

    def __init__(self):

       self.panel = []
       self.invalids = []


    @property
    def invalid_label(self):
        return('invalid')

    @property
    def size(self):
        return len(self.panel)


    def build(self, sample):
        """Use list of MIC values from test sample
        to initialize panel values in object

        Args:
            sample(list): List of MIC strings observed in panel

        Returns:
            list:
                list of panel value strings
                list of frequency counts

        """

        allmics = []
        record = Counter()
        i = 1
        for m in sample:
            try:
                mgpl = MGPL(m)         
            except ValueError as err:
                raise Exception('Unrecognized MIC format in element {}, {}.\n\t({})'.format(i, m, err))
            
            mval = str(mgpl)
            
            if not mval in record:
                allmics.append(mgpl)
            record[mval] += 1

            i += 1
        
        
        micvals = []
        for m in allmics:
            if not m.isna:
                micvals.append(m)
        micvals.sort()

        # Only the lowest and highest values should have signs
        if '>' in micvals[0].sign:
            raise ValueError('MIC value logic error in lowest mic: {}'.format(micvals[0]))

        if '<' in micvals[-1].sign:
            raise ValueError('MIC value logic error in highest mic: {}'.format(micvals[-1]))

        for m in micvals[1:-1]:
            if m.sign != '=':
                warnings.warn('MIC value logic error in middle mic: {}'.format(m))

        self.panel = micvals
        self.lookup = { str(val): idx for idx, val in enumerate(self.panel) }
        self.class_labels = [ str(c) for c in self.panel ]
        self.top_mgpl = micvals[-1]
        self.bottom_mgpl = micvals[0]

        counts = [ record[c] for c in self.class_labels ]

        return (self.class_labels, counts)


    def find(self, m):
        """Return ordered index of MIC value

        Args:
            m(str|int|float): MIC value, e.g. >=32, 2.0, <0.1

        Returns:
            list
                index(int): -1 if not found, or index in sorted panel list containing matching MIC value
                isna(bool):  True if MIC is a missing value e.g. '-' or 'NA'

        """

        # Convert to consistent representation
        mgpl = str(MGPL(m))

        if mgpl == 'NA':
            return (-1, True)
        if mgpl in self.lookup:
            return (self.lookup[mgpl], False)
        else:
            return (-1, False)


    def transform(self, m):
        """Return transformed MIC bin label

        Unlike regular 'find', if top value is has > symbol, and m is > than top value,
        method will return top value index, rather than -1.

        Similarly for the bottom value.

        Args:
            m: MGPL object

        Returns:
            list
                index(int): None if not found, or MIC bin string
                isna(bool):  True if MIC is a missing value e.g. '-' or 'NA'

        """

        # Convert to consistent representation
        try:
            mgpl = MGPL(m)
        except ValueError as err:
            mlabel = str(m)
            warnings.warn('Unrecognized MIC format: {}. Assigning as invalid'.format(mlabel))
            self.invalids.append(mlabel)
            return (self.invalid_label, False)
        mlabel = str(mgpl)

        if mlabel == 'NA':
            return (None, True)
        if mlabel in self.lookup:
            return (mlabel, False)
        else:
            # Encountered unrecognized MIC label

            # Is this greater than the top label?
            if (self.top_mgpl.sign == '>=' or self.top_mgpl.sign == '>') and mgpl > self.top_mgpl:
                return (str(self.top_mgpl), False)
            # Or less than bottom label?
            elif (self.bottom_mgpl.sign == '<=' or self.bottom_mgpl.sign == '<') and mgpl < self.bottom_mgpl:
                return (str(self.bottom_mgpl), False)
            # Or equal to top label?
            elif self.top_mgpl.sign == '>=' and self.top_mgpl.raw == mgpl.raw:
                return (str(self.top_mgpl), False)
            # Or equal to bottom label?
            elif self.bottom_mgpl.sign == '<=' and self.bottom_mgpl.raw == mgpl.raw:
                return (str(self.bottom_mgpl), False)
            else:
                # Something in between
                warnings.warn('Unknown MIC value: {}. Assigning as invalid'.format(m))
                self.invalids.append(mlabel)
                return(self.invalid_label, False)


    def map(self, mgpl):
        """Return transformed MIC bin label

        Unlike regular 'find', if top value has > symbol, and m is > than top value,
        method will return top value

        Similarly for the bottom value.

        This is different from 'transform' method which has a pre-defined panel. Anything that is a valid
        MIC between top and bottom values will be returned.

        Args:
            mgpl: MGPL object

        Returns:
            list
                index(int): None if not found, or MIC bin string
                isna(bool):  True if MIC is a missing value e.g. '-' or 'NA'

        """

        # Convert to consistent representation

        mlabel = str(mgpl)

        if mlabel == 'NA':
            return (None, True)
        else:
            # Valid MIC label

            # Is this greater than the top label?
            if (self.top_mgpl.sign == '>=' or self.top_mgpl.sign == '>') and mgpl > self.top_mgpl:
                return (str(self.top_mgpl), False)
            # Or less than bottom label?
            elif (self.bottom_mgpl.sign == '<=' or self.bottom_mgpl.sign == '<') and mgpl < self.bottom_mgpl:
                return (str(self.bottom_mgpl), False)
            # Or overlap with top label?
            elif self.top_mgpl.sign == '>=' and self.top_mgpl.raw == mgpl.raw:
                return (str(self.top_mgpl), False)
            # Or overlap bottom label?
            elif self.bottom_mgpl.sign == '<=' and self.bottom_mgpl.raw == mgpl.raw:
                return (str(self.bottom_mgpl), False)
            # Or equal to top label?
            elif self.top_mgpl == mgpl:
                return (str(self.top_mgpl), False)
            # Or equal to bottom label?
            elif self.bottom_mgpl == mgpl:
                return (str(self.bottom_mgpl), False)
            else:
                # Something in between

                if mgpl.sign != '=':
                    # In the internal range, all MICs must be specific values and not range symbols (i.e. >32)
                    return (self.invalid_label, False)
                else:
                    # Specific value between max and min
                    return (mlabel, False)


    def set_range(self, top, bottom):
        """Initialize ordered class labels for MICs in panel

        Top and bottom values define max and min MIC.
        Internal incongruent range MIC values (e.g. >8) will
        be assigned special class: invalid

        Args:
            top(str): Max panel value
            bottom(str): Min panel value

        Returns:
            None

        """

        top_mgpl = MGPL(top)
        top_label = str(top_mgpl)
        if top_mgpl.isna:
            raise ValueError('Invalid max MICPanel value {}'.format(top))
        if not top_label in self.lookup:
            warnings.warn('Max MICPanel value is not observed in the input MIC values: {}.'.format(top))
        self.top_mgpl = top_mgpl

        bottom_mgpl = MGPL(bottom)
        bottom_label = str(bottom_mgpl)
        if bottom_mgpl.isna:
            raise ValueError('Invalid min MICPanel value {}'.format(bottom))
        if not bottom_label in self.lookup:
            warnings.warn('Min MICPanel value is not observed in the input MIC values: {}.'.format(bottom))
        self.bottom_mgpl = bottom_mgpl

        self.class_mapping = {}
        self.class_labels = [bottom_label]

        for m in self.panel:
            mgpl = str(m)
            mlabel, isna = self.map(m)
            if not isna:
                if not mlabel in self.class_labels:
                    self.class_labels.append(mlabel)
                self.class_mapping[mgpl] = mlabel

        if not top_label in self.class_labels:
            self.class_labels.append(top_label)

        

class MGPL:
    """MGPL Class

    Represents one mg/L reading from MIC dilution

    e.g. <=0.15, 8, >32 are all valid MIC dilutions

    """

    def __init__(self, mic):

        sign, val, isna = self.parse(mic)
        if not isna and np.isnan(val):
            # Set nan as empty value (often these are empty cells)
            warnings.warn('Invalid MIC value: {}'.format(mic))
            sign = None
            val = None
            isna = True
        self._sign = sign
        self._raw = val
        self._isna = isna


    def __repr__(self):
        if self._isna:
            return 'NA'
        elif self._sign != '=':
            return self._sign + "{:01.4f}".format(self._raw)
        else:
            return "{:01.4f}".format(self._raw)


    # Specialized sorting functions
    # that assume > is only used on highest
    # and < is only is only used on lowest value
    def mycmp(self, other):

        if self.isna or other.isna:
            raise ValueError('MGPL sorting on empty values undefined')

        c = 0
        if self.raw < other.raw:
            c = -1
        elif self.raw > other.raw:
            c = 1

        if c == 0:
            if self.sign == '>=':
                if other.sign == '>':
                    return -1
                elif other.sign == '>=':
                    return 0
                else:
                    return 1
            elif self.sign == '>':
                if other.sign == '>':
                    return 0
                else:
                    return 1
            elif self.sign == '<=':
                if other.sign == '<':
                    return 1
                elif other.sign == '<=':
                    return 0
                else:
                    return -1
            elif self.sign == '<':
                if other.sign == '<':
                    return 0
                else:
                    return -1
            elif '>' in other.sign:
                return -1
            elif '<' in other.sign:
                return 1
        else:
            return c

    def __lt__(self, other):
        return self.mycmp(other) < 0
    def __gt__(self, other):
        return self.mycmp(other) > 0
    def __eq__(self, other):
        return self.mycmp(other) == 0
    def __le__(self, other):
        return self.mycmp(other) <= 0
    def __ge__(self, other):
        return self.mycmp(other) >= 0
    def __ne__(self, other):
        return self.mycmp(other) != 0


    @property
    def isna(self):
        return self._isna

    @property
    def sign(self):
        return self._sign

    @property
    def raw(self):
        return self._raw


    def parse(self, mic):
        """Parse string MIC to extract sign and float components

        Args:
            mic(string): e.g. >=32.0

        Returns list:
            (sign(string), value(float))
        """

        if mic == '-' or mic == 'NA':
            return (None, None, True)
        elif isinstance(mic, int):
            return ('=', float(mic), False)
        elif isinstance(mic, float):
            return ('=', mic, False)
        elif isinstance(mic, str):
            mic = mic.replace(' mg/L', '')
            sign = '='
            match = re.search(r'^(?P<sign>=|>=|<=|>|<|==)?\s*(?P<value>\d*\.\d+|\d+)', mic)
            if match:
                if match.group('sign'):
                    sign = match.group('sign')
                    if sign == '==':
                        sign = '='

                if match.group('value'):
                    value = float(match.group('value'))
                    if np.isnan(value):
                        raise ValueError('Invalid MIC. Cannot convert to float: {}'.format(mic))
                else:
                    raise ValueError('Invalid MIC. No value: {}'.format(mic))
            else:
                raise ValueError('Invalid MIC. Unrecognized format: {}'.format(mic))

            return (sign, value, False)

        else:
            raise ValueError("Unrecognized MIC type")



    




