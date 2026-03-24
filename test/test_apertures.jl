###############################################################################
# Tests for aperture geometry helpers
# Source: src/util/aperture_utils.jl
#
# Functions tested:
#   get_area(ap)           — area in pixels for Photometry.jl aperture/annulus objects
#   centroid_com(data[, mask]) — center of mass of a 2D array
###############################################################################

@testset "Aperture utilities" begin

    # =========================================================================
    # get_area  (analytic area formulas for each aperture type)
    # =========================================================================
    @testset "get_area" begin

        # CircularAperture(x, y, r) → area = π r²
        r = 3.0
        ap = CircularAperture(5.0, 5.0, r)
        @test Loki.get_area(ap) ≈ π * r^2 rtol=1e-12

        # EllipticalAperture(x, y, a, b, theta) → area = π a b
        a, b = 4.0, 2.0
        ap_ell = EllipticalAperture(5.0, 5.0, a, b, 0.0)
        @test Loki.get_area(ap_ell) ≈ π * a * b rtol=1e-12

        # RectangularAperture(x, y, w, h, theta) → area = w h
        w, h = 6.0, 4.0
        ap_rect = RectangularAperture(5.0, 5.0, w, h, 0.0)
        @test Loki.get_area(ap_rect) ≈ w * h rtol=1e-12

        # CircularAnnulus(x, y, r_in, r_out) → area = π(r_out² - r_in²)
        r_in, r_out = 2.0, 4.0
        ann_circ = CircularAnnulus(5.0, 5.0, r_in, r_out)
        @test Loki.get_area(ann_circ) ≈ π * (r_out^2 - r_in^2) rtol=1e-12

        # Circular annulus area = outer circle - inner circle
        @test Loki.get_area(ann_circ) ≈ Loki.get_area(CircularAperture(5.0,5.0,r_out)) -
                                         Loki.get_area(CircularAperture(5.0,5.0,r_in)) rtol=1e-12

        # EllipticalAnnulus(x, y, a_in, a_out, b_out, theta)
        # b_in is derived internally as a_in/a_out * b_out
        # area = π (a_out*b_out - a_in*b_in)
        a_in, a_out, b_out = 2.0, 4.0, 3.0
        b_in = a_in / a_out * b_out   # = 1.5
        ann_ell = EllipticalAnnulus(5.0, 5.0, a_in, a_out, b_out, 0.0)
        @test Loki.get_area(ann_ell) ≈ π * (a_out * b_out - a_in * b_in) rtol=1e-12

        # RectangularAnnulus(x, y, w_in, w_out, h_out, theta)
        # h_in is derived internally as h_out * w_in / w_out
        # area = w_out*h_out - w_in*h_in
        w_in, w_out, h_out = 2.0, 4.0, 6.0
        h_in = h_out * w_in / w_out   # = 3.0
        ann_rect = RectangularAnnulus(5.0, 5.0, w_in, w_out, h_out, 0.0)
        @test Loki.get_area(ann_rect) ≈ w_out * h_out - w_in * h_in rtol=1e-12

        # All areas are positive
        @test Loki.get_area(ap) > 0
        @test Loki.get_area(ap_ell) > 0
        @test Loki.get_area(ap_rect) > 0
        @test Loki.get_area(ann_circ) > 0
        @test Loki.get_area(ann_ell) > 0
        @test Loki.get_area(ann_rect) > 0
    end

    # =========================================================================
    # centroid_com(data[, mask])
    # Center-of-mass centroid: sums (index * value) over all unmasked pixels.
    # Returns 1-indexed coordinates.
    # =========================================================================
    @testset "centroid_com" begin

        # Uniform 3×3 array: centroid at geometric center (2.0, 2.0) in 1-based indexing
        data_uniform = ones(3, 3)
        c = Loki.centroid_com(data_uniform)
        @test c ≈ [2.0, 2.0] rtol=1e-10

        # Single nonzero pixel at [2, 3]: centroid = (2.0, 3.0)
        data_spike = zeros(5, 5)
        data_spike[2, 3] = 1.0
        c2 = Loki.centroid_com(data_spike)
        @test c2 ≈ [2.0, 3.0] rtol=1e-10

        # Symmetric array with two equal peaks: centroid at midpoint
        data_sym = zeros(1, 5)
        data_sym[1, 2] = 1.0
        data_sym[1, 4] = 1.0
        c3 = Loki.centroid_com(data_sym)
        @test c3[2] ≈ 3.0 rtol=1e-10   # midpoint between columns 2 and 4

        # Mask: masking all but one pixel → centroid at that pixel
        data = ones(4, 4)
        mask = trues(4, 4)
        mask[3, 2] = false   # unmasked pixel at (3, 2)
        c4 = Loki.centroid_com(data, mask)
        @test c4 ≈ [3.0, 2.0] rtol=1e-10

        # 1D array: centroid of [0, 0, 3, 0, 0] = index 3
        data_1d = reshape([0.0, 0.0, 3.0, 0.0, 0.0], 5, 1)
        c5 = Loki.centroid_com(data_1d)
        @test c5[1] ≈ 3.0 rtol=1e-10

        # No mask version == passing all-false mask
        c_no_mask  = Loki.centroid_com(data_uniform)
        c_all_unmasked = Loki.centroid_com(data_uniform, falses(3, 3))
        @test c_no_mask ≈ c_all_unmasked rtol=1e-12
    end

end
