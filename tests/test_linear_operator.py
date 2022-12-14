# Copyright 2022 The JaxLinOp Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pytest
from jaxlinop.linear_operator import LinearOperator


def test_covariance_operator() -> None:
    with pytest.raises(TypeError):
        LinearOperator()


class DummyLinearOperator(LinearOperator):
    def diagonal(self, *args, **kwargs):
        pass

    def shape(self, *args, **kwargs):
        pass

    def __mul__(self, *args, **kwargs):
        """Multiply linear operator by scalar."""
        pass

    def _add_diagonal(self, *args, **kwargs):
        pass

    def __matmul__(self, *args, **kwargs):
        """Matrix multiplication."""
        pass

    def to_dense(self, *args, **kwargs):
        pass

    @classmethod
    def from_dense(self, *args, **kwargs):
        pass


def test_can_instantiate() -> None:
    """Test if the covariance operator can be instantiated."""
    res = DummyLinearOperator()

    assert isinstance(res, DummyLinearOperator)
    assert res.name == "DummyLinearOperator"
