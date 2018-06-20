# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

import cirq


class MultiTargetCNot(cirq.CompositeGate, cirq.TextDiagrammableGate):
    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        if qubit_count is None:
            return '@', 'X'
        return ('@',) + tuple(['X'] * (qubit_count - 1))

    def default_decompose(self, qubits):
        return [cirq.CNOT(qubits[0], t) for t in qubits[1:]]

    def __repr__(self):
        return 'MultiTargetCNot'


class MultiTargetNotC(cirq.CompositeGate, cirq.TextDiagrammableGate):
    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        if qubit_count is None:
            return '⊕', 'Z'
        return ('#',) + tuple(['Z'] * (qubit_count - 1))

    def default_decompose(self, qubits):
        return [cirq.CNOT(t, qubits[0]) for t in qubits[1:]]

    def __repr__(self):
        return 'MultiTargetNotC'


MNot = MultiTargetCNot()
HMNot = MultiTargetNotC()


def single_errors(n):
    return [[k] for k in range(n)]


def double_errors(n):
    return [[k1, k2] for k1 in range(n) for k2 in range(k1 + 1, n)]


def check_block_code(k, bad_t_gates, reps):
    q1 = cirq.NamedQubit('$1')
    q2 = cirq.NamedQubit('$2')
    groups = [
        [
            cirq.NamedQubit(chr(97 + j) + str(i))
            for i in range(1, 5)
        ]
        for j in range(k + 2)
    ]

    t_id = 0

    def next_noisy_t(q, inverse=False):
        nonlocal t_id
        result = [cirq.T(q)**(-1 if inverse else 1)]
        if t_id in bad_t_gates:
            result.append(cirq.Z(q))
        t_id += 1
        return result

    output_qubits = [g[0] for g in groups[2:]]
    measured_qubits = [q1, q2] + [t
                                  for i in range(len(groups))
                                  for t in groups[i][0 if i < 2 else 1:]]

    circuit = cirq.Circuit.from_ops(
        # Initial H layer.
        cirq.H(q1),
        cirq.H(q2),
        cirq.H.on_each(g[0] for g in groups[1:]),

        # NOTC layer.
        [HMNot(*[g[0] for g in groups])],

        # CNOT layers.
        [MNot(*g) for g in groups],
        [MNot(q2, *[t for g in groups for t in [g[1], g[2]]])],
        [MNot(q1, *[t for g in groups for t in [g[0], g[2]]])],

        # Noisy T layer.
        next_noisy_t(groups[0][0], inverse=True),
        next_noisy_t(groups[1][0], inverse=True),
        [next_noisy_t(g[1], inverse=True) for g in groups],
        [next_noisy_t(t) for g in groups for t in [g[2], g[3]]],

        # Measure layer.
        cirq.H.on_each(measured_qubits),
        cirq.measure_each(measured_qubits),

        # Testing layer. Measures outputs in the T vs TZ basis.
        cirq.T.inverse().on_each(output_qubits),
        cirq.H.on_each(output_qubits),
        cirq.measure_each(output_qubits)
    )
    # print(circuit)

    # Simulate
    results = cirq.google.XmonSimulator().run(circuit, repetitions=reps)

    def read(*q: cirq.QubitId):
        result = np.zeros([reps], dtype=np.bool)
        for e in q:
            result ^= results.measurements[str(e)][:, 0]
        return result

    # Post-process
    parity = read(*[t for g in groups[:2] for t in g])
    outs = [read(g[0]) for g in groups[2:]]
    a = read(q1,
             groups[0][2], groups[0][0],
             groups[1][2], groups[1][0])
    b = read(q2,
             groups[0][1], groups[0][2],
             groups[1][1], groups[1][2])
    for t in groups[0]:
        for i in range(k):
            outs[i] ^= read(t)
    for i in range(k):
        g = groups[i + 2]
        for t in g[1:]:
            outs[i] ^= read(t)
        a ^= read(g[1], g[3])
        b ^= read(g[1], g[2])

    # print(results)
    # print('check1 ($1)', _bitstring(a))
    # print('check2 ($2)', _bitstring(b))
    # print('check3 (ab parity)', _bitstring(parity))
    # for i, out in enumerate(outs):
    #     print('out', i, _bitstring(out))

    detected_errors = a | b | parity
    any_actual_errors = read()
    for out in outs:
        any_actual_errors |= out
    missed_errors = any_actual_errors & ~detected_errors

    if np.any(missed_errors):
        print('bad case',
              bad_t_gates,
              'check bits',
              _bitstring(a),
              _bitstring(b),
              _bitstring(parity),
              'outs',
              [_bitstring(out) for out in outs])
    return np.any(detected_errors), np.any(any_actual_errors)


def _bitstring(x):
    return ''.join('1' if e else '0' for e in x)


def main():
    k = 2
    n = 8 + 3*k
    reps = 10

    print('perfect T gates case')
    detected, actual = check_block_code(k, [], reps)
    assert not detected and not actual
    print('-')

    print('single error cases')
    for s in single_errors(n):
        detected, actual = check_block_code(k, s, reps)
        assert detected
    print('-')

    print('double error cases')
    count = 0
    for s in double_errors(n):
        detected, actual = check_block_code(k, s, reps)
        if not detected and actual:
            count += 1
    print('-')

    print('total undetected doubles', count)


if __name__ == '__main__':
    main()
