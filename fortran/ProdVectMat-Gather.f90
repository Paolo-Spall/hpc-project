program vect_mat_mul
    USE MPI
    implicit none
    
    integer, parameter :: K2 = selected_real_kind(15, 30)
    integer :: n,i,j,k,st, remainder, disp
    integer :: size, ierr, ecode, rank , status(MPI_STATUS_SIZE)
    real :: cpu1, cpu2, deltat
    real(KIND=K2), allocatable, dimension(:,:) :: M
    real(KIND=K2), allocatable, dimension(:) :: V, C, buffer
    integer, allocatable, dimension(:) :: DISPLS, RECVCOUNTS
    integer :: Ncolumns
    character(len=100) :: times_filename

    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD,size,ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD,rank,ierr)
    
    CALL CPU_TIME(cpu1)

    print*, 'MPI initialized. Rank:', rank, 'of', size

    OPEN(11,FILE="input_n.txt",STATUS="OLD",IOSTAT=st)
    READ(11,*,IOSTAT=st) n
    CLOSE(11, STATUS='KEEP')

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
    
    ! do i=1,n
    !     print*, M(i,:)
    ! end do

    ! initialize resulting vector
    allocate(C(n))

    C = MATMUL(V, M)

    !communication part
    allocate(buffer(n))
    call MPI_GATHERV(C, Ncolumns, MPI_DOUBLE_PRECISION, buffer, RECVCOUNTS, DISPLS, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)
    if (rank == 0) then
        ! The root process gathers the results
        ! print '(A,10(E15.7E3,1X))', 'Middle ten gathered results:', buffer(496:505)
        ! print*, 'Last five elements of the gathered vector:', buffer(n-4:n)
        OPEN(12,FILE="ProdVectMat-Gather_f90.output",STATUS="REPLACE")
        do i=1, 100
            WRITE(12,*) buffer(i)
        end do
        CLOSE(12, STATUS='KEEP')
    end if

    CALL CPU_TIME(cpu2)

    deltat = cpu2 - cpu1
    if (rank == 0) then
        WRITE(times_filename, '(A,I0,A)') 'times_', n, '_ProdVectMat-Gather_f90.txt'
        OPEN(13,FILE=times_filename,status='unknown', position='append')
        WRITE(13, '(I2, X, F10.6)') size, deltat
        CLOSE(13, STATUS='KEEP')
        !print*, 'Time taken for initialization:', deltat
    end if

    CALL MPI_FINALIZE(ierr)
end program vect_mat_mul

