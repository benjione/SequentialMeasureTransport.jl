import SequentialMeasureTransport.Hermitian_to_low_vec
import SequentialMeasureTransport.low_vec_to_Symmetric
import SequentialMeasureTransport.view_mat_for_to_symmetric


@testset "specific test" begin
    A = [1 2 3; 2 4 5; 3 5 6]
    @test Hermitian_to_low_vec(A) == [1, 2, 3, 4, 5, 6]

    @test low_vec_to_Symmetric([1, 2, 3, 4, 5, 6]) == A
end

@testset "closed test" begin
    for i in 1:100
        N = rand(10:200)
        A = rand(N, N)
        A = Hermitian(A)
        @test low_vec_to_Symmetric(Hermitian_to_low_vec(A)) == A
    end
end

@testset "view mat test" begin
    for i in 1:100
        N = rand(10:200)
        A = rand(N, N)
        v_mat = view_mat_for_to_symmetric(N)
        A = Hermitian(A)
        @test low_vec_to_Symmetric(Hermitian_to_low_vec(A), v_mat) == A
    end
end
