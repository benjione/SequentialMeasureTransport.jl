using SequentialMeasureTransport: DownwardClosedTensorizer, 
            Tensorizer, next_index_proposals, add_index!


@testset "Downward closed tensorizer" begin
    @testset "2D creation" begin
        t = DownwardClosedTensorizer(2, 2)
        @test issetequal(t.index_list, Tuple.([[1, 1], [1, 2], [2, 1], 
                               [1, 3], [2, 2], [3, 1]]))
        @test issetequal(t.M_inner, Tuple.([[1,3], [2, 2], [3, 1]]))
    end

    @testset "2D proposals" begin
        t = DownwardClosedTensorizer(2, 2)
        prop = next_index_proposals(t)
        @test issetequal(prop, Tuple.([[4, 1], [3, 2], [2, 3], 
                               [1, 4]]))
    end

    @testset "2D add index" begin
        t = DownwardClosedTensorizer(2, 2)
        add_index!(t, (4, 1))
        @test issetequal(t.index_list, Tuple.([[1, 1], [1, 2], [2, 1], 
                               [1, 3], [2, 2], [3, 1], [4, 1]]))
        @test issetequal(t.M_inner, Tuple.([[1,3], [2, 2], [3, 1],[4, 1]]))
    end

    @testset "3D creation" begin
        t = DownwardClosedTensorizer(3, 2)
        @test issetequal(t.index_list, Tuple.([[1, 1, 1], [1, 1, 2], [1, 2, 1], 
                               [2, 1, 1], [1, 2, 2], [2, 1, 2], 
                               [2, 2, 1], [1, 1, 3], 
                               [1, 3, 1], [3, 1, 1]]))
        @test issetequal(t.M_inner, Tuple.([[1, 1, 3], [2, 2, 1], [1,2,2],
                        [2, 1, 2],
                        [1, 3, 1], [3, 1, 1]]))
    end

    @testset "3D proposals" begin
        t = DownwardClosedTensorizer(3, 2)
        prop = next_index_proposals(t)
        @test issetequal(prop, Tuple.([[1,1,4], [1,2,3], [1,3,2], [1,4,1], 
                               [2,1,3], [2,2,2], [2,3,1], 
                               [3,1,2], [3,2,1], [4,1,1]]))
    end
end