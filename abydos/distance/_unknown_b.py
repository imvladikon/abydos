# Copyright 2019-2020 by Christopher C. Little.
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

"""abydos.distance._unknown_b.

Unknown B similarity
"""

from typing import Any, Counter as TCounter, Optional, Sequence, Set, Union

from abydos.distance._token_distance import _TokenDistance
from abydos.tokenizer import _Tokenizer

__all__ = ['UnknownB']


class UnknownB(_TokenDistance):
    r"""Unknown B similarity.

    For two sets X and Y and a population N, Unknown B similarity, which
    :cite:`Morris:2012` attributes to :cite:`Doolittle:1884` but could not be
    located in that source, is

        .. math::

            sim_{UnknownB}(X, Y) =
            \frac{(|X \cap Y| \cdot |(N \setminus X) \setminus Y| -
            |X \setminus Y| \cdot |Y \setminus X|)^2}
            {|X| \cdot |Y| \cdot |N \setminus X| \cdot |N \setminus Y|}

    In :ref:`2x2 confusion table terms <confusion_table>`, where a+b+c+d=n,
    this is

        .. math::

            sim_{UnknownB} =
            \frac{(ad-bc)^2}{(a+b)(a+c)(b+d)(c+d)}

    .. versionadded:: 0.4.0
    """

    def __init__(
        self,
        alphabet: Optional[
            Union[TCounter[str], Sequence[str], Set[str], int]
        ] = None,
        tokenizer: Optional[_Tokenizer] = None,
        intersection_type: str = 'crisp',
        **kwargs: Any
    ) -> None:
        """Initialize UnknownB instance.

        Parameters
        ----------
        alphabet : Counter, collection, int, or None
            This represents the alphabet of possible tokens.
            See :ref:`alphabet <alphabet>` description in
            :py:class:`_TokenDistance` for details.
        tokenizer : _Tokenizer
            A tokenizer instance from the :py:mod:`abydos.tokenizer` package
        intersection_type : str
            Specifies the intersection type, and set type as a result:
            See :ref:`intersection_type <intersection_type>` description in
            :py:class:`_TokenDistance` for details.
        **kwargs
            Arbitrary keyword arguments

        Other Parameters
        ----------------
        qval : int
            The length of each q-gram. Using this parameter and tokenizer=None
            will cause the instance to use the QGram tokenizer with this
            q value.
        metric : _Distance
            A string distance measure class for use in the ``soft`` and
            ``fuzzy`` variants.
        threshold : float
            A threshold value, similarities above which are counted as
            members of the intersection for the ``fuzzy`` variant.


        .. versionadded:: 0.4.0

        """
        super(UnknownB, self).__init__(
            alphabet=alphabet,
            tokenizer=tokenizer,
            intersection_type=intersection_type,
            **kwargs
        )

    def sim(self, src: str, tar: str) -> float:
        """Return the Unknown B similarity of two strings.

        Parameters
        ----------
        src : str
            Source string (or QGrams/Counter objects) for comparison
        tar : str
            Target string (or QGrams/Counter objects) for comparison

        Returns
        -------
        float
            Unknown B similarity

        Examples
        --------
        >>> cmp = UnknownB()
        >>> cmp.sim('cat', 'hat')
        0.24744247205785666
        >>> cmp.sim('Niall', 'Neil')
        0.13009912077202224
        >>> cmp.sim('aluminum', 'Catalan')
        0.011710186806836291
        >>> cmp.sim('ATCG', 'TAGC')
        4.1196952743799446e-05


        .. versionadded:: 0.4.0

        """
        if src == tar:
            return 1.0

        self._tokenize(src, tar)

        a = self._intersection_card()
        b = self._src_only_card()
        c = self._tar_only_card()
        d = self._total_complement_card()

        num = (a * d - b * c) ** 2
        if num:
            return num / ((a + b) * (a + c) * (b + d) * (c + d))
        return 0.0


if __name__ == '__main__':
    import doctest

    doctest.testmod()
