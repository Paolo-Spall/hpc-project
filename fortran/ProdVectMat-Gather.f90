program vect_mat_mul
    USE MPI
    implicit none
    
    integer :: n,i,j,k,st, remainder, disp
    integer :: size, ierr, ecode, rank , status(MPI_STATUS_SIZE)
    real :: cpu1, cpu2, deltat
    real, allocatable, dimension(:,:) :: M
    real, allocatable, dimension(:) :: V, C, buffer
    integer, allocatable, dimension(:) :: DISPLS, RECVCOUNTS
    integer :: Ncolumns

    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD,size,ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD,rank,ierr)
    
    CALL CPU_TIME(cpu1)

    print*, 'MPI initialized. Rank:', rank, 'of', size

    OPEN(11,FILE="input_n.txt",STATUS="OLD",IOSTAT=st)
    READ(11,*,IOSTAT=st) n

    print*, 'Dimension of the matrix:', n

    !computing number of columns that each process will handle
    print*, 'Number of processes:', size
    Ncolumns = n / size
    remainder = mod(n, size)
    print*, 'Number of columns per process:', Ncolumns

    ! Setting DISPLS and RECVCOUNTS for MPI_GATHER
    allocate(DISPLS(size))
    allocate(RECVCOUNTS(size))

    disp = 0
    do i=0, size-1
        DISPLS(i+1) = disp
        if (i < remainder) then
            RECVCOUNTS(i+1) = Ncolumns + 1
        else
            RECVCOUNTS(i+1) = Ncolumns
        end if
        disp = disp + RECVCOUNTS(i+1)
    end do

    print*, 'DISPLS:', DISPLS
    print*, 'RECVCOUNTS:', RECVCOUNTS
    !if 

    ! If the number of columns is not divisible by size increase by one
    ! the first elements
   
    if (rank < remainder) then
        Ncolumns = Ncolumns + 1
    end if

    print*, 'Adjusted number of columns for process', rank, ':', Ncolumns

    ! Initialize Vector
    allocate(V(n))   
    
    do i=1,n
        V(i) = i
    end do

    ! Initialize Matrix
    allocate(M(n,Ncolumns))

    do i=1,n
        do j = 1, Ncolumns
        !do j=DISPLS(rank)+1, DISPLS(rank)+1 + RECVCOUNTS(rank)
            M(i, j) = DISPLS(rank+1) + j
        end do
    end do
    
    do i=1,n
        print*, M(i,:)
    end do

    ! initialize resulting vector
    allocate(C(n))

    C = MATMUL(M,V)

    CALL CPU_TIME(cpu2)

    deltat = cpu2 - cpu1

    print*, 'Time taken for initialization:', deltat

    CALL MPI_FINALIZE(ierr)
end program vect_mat_mul

