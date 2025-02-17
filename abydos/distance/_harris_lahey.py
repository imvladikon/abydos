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

"""abydos.distance._harris_lahey.

Harris & Lahey similarity
"""

from typing import Any, Counter as TCounter, Optional, Sequence, Set, Union

from abydos.distance._token_distance import _TokenDistance
from abydos.tokenizer import _Tokenizer

__all__ = ['HarrisLahey']


class HarrisLahey(_TokenDistance):
    r"""Harris & Lahey similarity.

    For two sets X and Y and a population N, Harris & Lahey similarity
    :cite:`Harris:1978` is

        .. math::

            sim_{HarrisLahey}(X, Y) =
            \frac{|X \cap Y|}{|X \cup Y|}\cdot
            \frac{|N \setminus Y| + |N \setminus X|}{2|N|}+
            \frac{|(N \setminus X) \setminus Y|}{|N \setminus (X \cap Y)|}\cdot
            \frac{|X| + |Y|}{2|N|}

    In :ref:`2x2 confusion table terms <confusion_table>`, where a+b+c+d=n,
    this is

        .. math::

            sim_{HarrisLahey} =
            \frac{a}{a+b+c}\cdot\frac{2d+b+c}{2n}+
            \frac{d}{d+b+c}\cdot\frac{2a+b+c}{2n}

    Notes
    -----
    Most catalogs of similarity coefficients
    :cite:`Warrens:2008,Morris:2012,Xiang:2013` omit the :math:`n` terms in the
    denominators, but the worked example in :cite:`Harris:1978` makes it clear
    that this is intended in the original.

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
        """Initialize HarrisLahey instance.

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
        super(HarrisLahey, self).__init__(
            alphabet=alphabet,
            tokenizer=tokenizer,
            intersection_type=intersection_type,
            **kwargs
        )

    def sim(self, src: str, tar: str) -> float:
        """Return the Harris & Lahey similarity of two strings.

        Parameters
        ----------
        src : str
            Source string (or QGrams/Counter objects) for comparison
        tar : str
            Target string (or QGrams/Counter objects) for comparison

        Returns
        -------
        float
            Harris & Lahey similarity

        Examples
        --------
        >>> cmp = HarrisLahey()
        >>> cmp.sim('cat', 'hat')
        0.3367085964820711
        >>> cmp.sim('Niall', 'Neil')
        0.22761577457069784
        >>> cmp.sim('aluminum', 'Catalan')
        0.07244410503054725
        >>> cmp.sim('ATCG', 'TAGC')
        0.006296204706372345


        .. versionadded:: 0.4.0

        """
        if src == tar:
            return 1.0

        self._tokenize(src, tar)

        a = self._intersection_card()
        b = self._src_only_card()
        c = self._tar_only_card()
        d = self._total_complement_card()
        n = self._population_unique_card()

        score = 0.0
        if a and (d + b + c):
            score += a / (a + b + c) * (2 * d + b + c) / (2 * n)
        if d and (a + b + c):
            score += d / (d + b + c) * (2 * a + b + c) / (2 * n)
        return score


if __name__ == '__main__':
    import doctest

    doctest.testmod()
