# GT4Py New Semantic Model - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.  GT4Py
# New Semantic Model is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or any later version.
# See the LICENSE.txt file at the top-level directory of this distribution for
# a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import math

import numpy as np
from atlas4py import (
    Config,
    StructuredGrid,
    StructuredMeshGenerator,
    Topology,
    build_edges,
    build_median_dual_mesh,
    build_node_to_edge_connectivity,
    functionspace,
)


def assert_close(expected, actual):
    assert math.isclose(expected, actual), "expected={}, actual={}".format(expected, actual)


class nabla_setup:
    @staticmethod
    def _default_config():
        config = Config()
        config["triangulate"] = True
        config["angle"] = 20.0
        return config

    def __init__(self, *, grid=StructuredGrid("O32"), config=None):
        if config is None:
            config = self._default_config()
        mesh = StructuredMeshGenerator(config).generate(grid)

        fs_edges = functionspace.EdgeColumns(mesh, halo=1)
        fs_nodes = functionspace.NodeColumns(mesh, halo=1)

        build_edges(mesh)
        build_node_to_edge_connectivity(mesh)
        build_median_dual_mesh(mesh)

        edges_per_node = max(
            [mesh.nodes.edge_connectivity.cols(node) for node in range(0, fs_nodes.size)]
        )

        self.mesh = mesh
        self.fs_edges = fs_edges
        self.fs_nodes = fs_nodes
        self.edges_per_node = edges_per_node

    @property
    def edges2node_connectivity(self):
        return self.mesh.edges.node_connectivity

    @property
    def nodes2edge_connectivity(self):
        return self.mesh.nodes.edge_connectivity

    @property
    def nodes_size(self):
        return self.fs_nodes.size

    @property
    def edges_size(self):
        return self.fs_edges.size

    @staticmethod
    def _is_pole_edge(e, edge_flags):
        return Topology.check(edge_flags[e], Topology.POLE)

    @property
    def is_pole_edge_field(self):
        edge_flags = np.array(self.mesh.edges.flags())

        pole_edge_field = np.zeros((self.edges_size,))
        for e in range(self.edges_size):
            pole_edge_field[e] = self._is_pole_edge(e, edge_flags)
        return edge_flags

    @property
    def sign_field(self):
        node2edge_sign = np.zeros((self.nodes_size, self.edges_per_node))
        edge_flags = np.array(self.mesh.edges.flags())

        for jnode in range(0, self.nodes_size):
            node_edge_con = self.mesh.nodes.edge_connectivity
            edge_node_con = self.mesh.edges.node_connectivity
            for jedge in range(0, node_edge_con.cols(jnode)):
                iedge = node_edge_con[jnode, jedge]
                ip1 = edge_node_con[iedge, 0]
                if jnode == ip1:
                    node2edge_sign[jnode, jedge] = 1.0
                else:
                    node2edge_sign[jnode, jedge] = -1.0
                    if self._is_pole_edge(iedge, edge_flags):
                        node2edge_sign[jnode, jedge] = 1.0
        return node2edge_sign

    @property
    def S_fields(self):
        S = np.array(self.mesh.edges.field("dual_normals"), copy=False)
        S_MXX = np.zeros((self.edges_size))
        S_MYY = np.zeros((self.edges_size))

        MXX = 0
        MYY = 1

        rpi = 2.0 * math.asin(1.0)
        radius = 6371.22e03
        deg2rad = 2.0 * rpi / 360.0

        for i in range(0, self.edges_size):
            S_MXX[i] = S[i, MXX] * radius * deg2rad
            S_MYY[i] = S[i, MYY] * radius * deg2rad

        assert math.isclose(min(S_MXX), -103437.60479272791)
        assert math.isclose(max(S_MXX), 340115.33913622628)
        assert math.isclose(min(S_MYY), -2001577.7946404363)
        assert math.isclose(max(S_MYY), 2001577.7946404363)

        return S_MXX, S_MYY

    @property
    def vol_field(self):
        rpi = 2.0 * math.asin(1.0)
        radius = 6371.22e03
        deg2rad = 2.0 * rpi / 360.0
        vol_atlas = np.array(self.mesh.nodes.field("dual_volumes"), copy=False)
        # dual_volumes 4.6510228700066421    68.891611253882218    12.347560975609632
        assert_close(4.6510228700066421, min(vol_atlas))
        assert_close(68.891611253882218, max(vol_atlas))

        vol = np.zeros((vol_atlas.size))
        for i in range(0, vol_atlas.size):
            vol[i] = vol_atlas[i] * pow(deg2rad, 2) * pow(radius, 2)
        # VOL(min/max):  57510668192.214096    851856184496.32886
        assert_close(57510668192.214096, min(vol))
        assert_close(851856184496.32886, max(vol))
        return vol

    @property
    def input_field(self):
        klevel = 0
        MXX = 0
        MYY = 1
        rpi = 2.0 * math.asin(1.0)
        radius = 6371.22e03
        deg2rad = 2.0 * rpi / 360.0

        zh0 = 2000.0
        zrad = 3.0 * rpi / 4.0 * radius
        zeta = rpi / 16.0 * radius
        zlatc = 0.0
        zlonc = 3.0 * rpi / 2.0

        m_rlonlatcr = self.fs_nodes.create_field(
            name="m_rlonlatcr",
            levels=1,
            dtype=np.float64,
            variables=self.edges_per_node,
        )
        rlonlatcr = np.array(m_rlonlatcr, copy=False)

        m_rcoords = self.fs_nodes.create_field(
            name="m_rcoords", levels=1, dtype=np.float64, variables=self.edges_per_node
        )
        rcoords = np.array(m_rcoords, copy=False)

        m_rcosa = self.fs_nodes.create_field(name="m_rcosa", levels=1, dtype=np.float64)
        rcosa = np.array(m_rcosa, copy=False)

        m_rsina = self.fs_nodes.create_field(name="m_rsina", levels=1, dtype=np.float64)
        rsina = np.array(m_rsina, copy=False)

        m_pp = self.fs_nodes.create_field(name="m_pp", levels=1, dtype=np.float64)
        rzs = np.array(m_pp, copy=False)

        rcoords_deg = np.array(self.mesh.nodes.field("lonlat"))

        for jnode in range(0, self.nodes_size):
            for i in range(0, 2):
                rcoords[jnode, klevel, i] = rcoords_deg[jnode, i] * deg2rad
                rlonlatcr[jnode, klevel, i] = rcoords[jnode, klevel, i]  # This is not my pattern!
            rcosa[jnode, klevel] = math.cos(rlonlatcr[jnode, klevel, MYY])
            rsina[jnode, klevel] = math.sin(rlonlatcr[jnode, klevel, MYY])
        for jnode in range(0, self.nodes_size):
            zlon = rlonlatcr[jnode, klevel, MXX]
            zdist = math.sin(zlatc) * rsina[jnode, klevel] + math.cos(zlatc) * rcosa[
                jnode, klevel
            ] * math.cos(zlon - zlonc)
            zdist = radius * math.acos(zdist)
            rzs[jnode, klevel] = 0.0
            if zdist < zrad:
                rzs[jnode, klevel] = rzs[jnode, klevel] + 0.5 * zh0 * (
                    1.0 + math.cos(rpi * zdist / zrad)
                ) * math.pow(math.cos(rpi * zdist / zeta), 2)

        assert_close(0.0000000000000000, min(rzs))
        assert_close(1965.4980340735883, max(rzs))
        return rzs[:, klevel]
