program vect_mat_mul
    USE MPI
    implicit none
    
    ! defining double precision tipe
    integer, parameter :: K2 = selected_real_kind(15, 30)
    ! declaring indexes and integer aux variables
    integer :: n,i,j,k, Ncolumns, st, remainder, disp, n_print, n_elements
    ! decl MPI variables
    integer :: size, ierr, ecode, rank , status(MPI_STATUS_SIZE)
    real(KIND=K2) :: cpu1, cpu2, deltat
    ! declaring matrix and vector variables
    real(KIND=K2), allocatable, dimension(:,:) :: M
    real(KIND=K2), allocatable, dimension(:) :: V, C, buffer
    integer, allocatable, dimension(:) :: DISPLS, RECVCOUNTS
    ! filename strings
    character(len=100) :: times_filename

    ! initializing parallel part of the code
    call MPI_INIT(ierr) ! global communicator
    call MPI_COMM_SIZE(MPI_COMM_WORLD,size,ierr) !extracting task id
    call MPI_COMM_RANK(MPI_COMM_WORLD,rank,ierr)!extracting number of processes
    
    !starting time measuring
    cpu1 = MPI_WTIME()

    ! Read the size of the matrix from a file
    OPEN(11,FILE="input_n.txt",STATUS="OLD",IOSTAT=st)
    READ(11,*,IOSTAT=st) n
    CLOSE(11, STATUS='KEEP')

    !computing number of columns that each process will handle
    Ncolumns = n / size
    remainder = mod(n, size)

    ! Setting DISPLS and RECVCOUNTS for MPI_GATHER
    allocate(DISPLS(size)) !width of resulting vector for each process
    allocate(RECVCOUNTS(size)) ! index of the first element of the local vector in the global one

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

    ! If the number of columns is not divisible by size increase by one
    ! the first x elements, with x being the reminder of n/size
   
    if (rank < remainder) then
        Ncolumns = Ncolumns + 1
    end if

    ! Initialize Vector as [ 1,2,...,n ]
    allocate(V(n))   
    
    do i=1,n
        V(i) = i
    end do

    ! Initialize Matrix with n rows, each one like [ 1,2,...,n ]
    allocate(M(n,Ncolumns))

    do i=1,n
        do j = 1, Ncolumns
            M(i, j) = DISPLS(rank+1) + j
        end do
    end do

    ! initialize resulting vector
    allocate(C(Ncolumns))

    !Computing the vector-matrix product
    do j =1, Ncolumns
        C(j) = 0.0
        do k=1, n
            C(j) = C(j) + M(k,j) * V(k)
        end do
    end do

    ! COMMUNICATION AND SAVING DATA SECTION

    ! saving the fisrts 100 elemnts to the input file
    ! rank 0 receive all resulting C vectors from others ranks
    if (rank == 0) then
        n_print = MIN(n, 100) ! number of elements to be printed
        
        ! saving the fisrts 100 elemnts to the input file
        OPEN(12,FILE="ProdVectMat-Receive-Cycle_f90.output",STATUS="REPLACE")
        do i=1, min(n_print, Ncolumns)
            WRITE(12,*) C(i)
            n_print = n_print -1
        end do
        
        !iterating over the tasks to receive receive
        do i=1, size-1
            n_elements = RECVCOUNTS(i+1)
            allocate(buffer(n_elements))! initialize receive buffer
            ! receiving the C i-th vector into buffer
            call MPI_RECV(buffer, n_elements, MPI_DOUBLE_PRECISION, i, 10, MPI_COMM_WORLD, status, ierr)
            !writing the elements if still needed
            if (n_print > 0) then
                do j=1, min(n_print, n_elements)
                    WRITE(12,*) buffer(j)
                    n_print = n_print - 1
                end do
            end if
            deallocate(buffer)
        end do
        CLOSE(12, STATUS='KEEP')
    else
        ! for tasks 1,...,n sending the C vector to rank 0
        call MPI_SEND(C, Ncolumns, MPI_DOUBLE_PRECISION, 0, 10, MPI_COMM_WORLD, ierr)
    end if

    ! stop time measurment
    cpu2 = MPI_WTIME()

    ! saving computing time into correspondent times output file 
    deltat = cpu2 - cpu1
    if (rank == 0) then
        WRITE(times_filename, '(A,I0,A)') 'times_', n, '_ProdVectMat-Receive-Cycle_f90.txt'
        OPEN(13,FILE=times_filename,status='unknown', position='append')
        WRITE(13, '(I2, X, F10.6)') size, deltat
        CLOSE(13, STATUS='KEEP')
    end if

    ! finalizing parallel part of the code
    CALL MPI_FINALIZE(ierr)
end program vect_mat_mul

