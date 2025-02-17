# Copyright 2018-2020 by Christopher C. Little.
# This file is part of Abydos.
#
# Abydos is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Abydos is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Abydos. If not, see <http://www.gnu.org/licenses/>.

"""abydos.tokenizer._whitespace.

Whitespace tokenizer
"""

from typing import Callable, Optional, Union

from abydos.tokenizer._regexp import RegexpTokenizer

__all__ = ['WhitespaceTokenizer']


class WhitespaceTokenizer(RegexpTokenizer):
    """A whitespace tokenizer.

    Examples
    --------
    >>> WhitespaceTokenizer().tokenize('a b c f a c g e a b')
    WhitespaceTokenizer({'a': 3, 'b': 2, 'c': 2, 'f': 1, 'g': 1, 'e': 1})


    .. versionadded:: 0.4.0

    """

    def __init__(
        self,
        scaler: Optional[Union[str, Callable[[float], float]]] = None,
        flags: int = 0,
    ) -> None:
        """Initialize tokenizer.

        Parameters
        ----------
        scaler : None, str, or function
            A scaling function for the Counter:

                - None : no scaling
                - 'set' : All non-zero values are set to 1.
                - 'length' : Each token has weight equal to its length.
                - 'length-log' : Each token has weight equal to the log of its
                   length + 1.
                - 'length-exp' : Each token has weight equal to e raised to its
                   length.
                - a callable function : The function is applied to each value
                  in the Counter. Some useful functions include math.exp,
                  math.log1p, math.sqrt, and indexes into interesting integer
                  sequences such as the Fibonacci sequence.
        flags : int
            Flags to pass to the regular expression matcher. See the
            `documentation on Python's re module
            <https://docs.python.org/3/library/re.html#re.A>`_ for details.


        .. versionadded:: 0.4.0

        """
        super(WhitespaceTokenizer, self).__init__(
            scaler, regexp=r'\S+', flags=flags
        )


if __name__ == '__main__':
    import doctest

    doctest.testmod()
