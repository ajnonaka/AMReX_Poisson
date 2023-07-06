#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <AMReX_BCUtil.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MultiFabUtil.H>

#include "myfunc.H"

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
    return 0;
}

void main_main ()
{
    // What time is it now?  We'll use this to compute total run time.
    auto strt_time = amrex::second();

    // number of cells on each side of the domain
    int n_cell_x = 64;
    int n_cell_y = 64;
    int n_cell_z = 64;

    // dimensions of each box (or grid)
    Real prob_lo_x = 0.;
    Real prob_lo_y = 0.;
    Real prob_lo_z = 0.;
    Real prob_hi_x = 1.;
    Real prob_hi_y = 1.;
    Real prob_hi_z = 1.;

    // This is the largest size a grid can be
    int max_grid_size = 32;

    // inputs parameters
    {
        // ParmParse is way of reading inputs from the inputs file
        ParmParse pp;

        // We need to get n_cell from the inputs file - this is the number of cells on each side of
        //   a square (or cubic) domain.
        pp.query("n_cell_x",n_cell_x);
        pp.query("n_cell_y",n_cell_y);
        pp.query("n_cell_z",n_cell_z);

        // We need to query prob_lo_x/y/z from the inputs file - tlos is the physical dimensions of the domain
        pp.query("prob_lo_x",prob_lo_x);
        pp.query("prob_lo_y",prob_lo_y);
        pp.query("prob_lo_z",prob_lo_z);

        // We need to query prob_hi_x/y/z from the inputs file - this is the physical dimensions of the domain
        pp.query("prob_hi_x",prob_hi_x);
        pp.query("prob_hi_y",prob_hi_y);
        pp.query("prob_hi_z",prob_hi_z);

        // The domain is broken into boxes of size max_grid_size
        pp.query("max_grid_size",max_grid_size);
    }

    // assume periodic in all directions
    Vector<int> is_periodic(AMREX_SPACEDIM,1);

    // make BoxArray and Geometry
    BoxArray ba;
    Geometry geom;
    {
        IntVect dom_lo(AMREX_D_DECL(         0,          0,          0));
        IntVect dom_hi(AMREX_D_DECL(n_cell_x-1, n_cell_y-1, n_cell_z-1));
        Box domain(dom_lo, dom_hi);

        // Initialize the boxarray "ba" from the single box "bx"
        ba.define(domain);
        // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
        ba.maxSize(max_grid_size);

       // This defines the physical box, [-1,1] in each direction.
        RealBox real_box({AMREX_D_DECL(prob_lo_x,prob_lo_y,prob_lo_z)},
                         {AMREX_D_DECL(prob_hi_x,prob_hi_y,prob_hi_z)});

        // This defines a Geometry object
        geom.define(domain,&real_box,CoordSys::cartesian,is_periodic.data());
    }

    // Nghost = number of ghost cells for each array
    int Nghost = 1;

    // Ncomp = number of components for each array
    int Ncomp  = 1;

    // How Boxes are distrubuted among MPI processes
    DistributionMapping dmap(ba);

    // we allocate two phi multifabs; one will store the old state, the other the new.
    MultiFab phi(ba, dmap, Ncomp, Nghost);
    MultiFab rhs(ba, dmap, Ncomp, 0);

    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // set initial guess for solution phi to zero
    phi.setVal(0.);

    // initialize rhs
    for (MFIter mfi(rhs); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        auto const& rhs_ptr = rhs.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            // FIXME - replace -1.0 with prob_lo[0] and prob_lo[1]
            Real x = -1.0 + (i+0.5)*dx[0];
            Real y = -1.0 + (j+0.5)*dx[1];

            Real z = (AMREX_SPACEDIM == 3) ? -1.0 + (k+0.5)*dx[2] : 0.;
            
            rhs_ptr(i,j,k) = std::exp(-10.*(x*x+y*y+z*z));
        });
    }

    // assortment of solver and parallization options and parameters
    // see AMReX_MLLinOp.H for the defaults, accessors, and mutators
    LPInfo info;

    // Implicit solve using MLABecLaplacian class
    MLABecLaplacian mlabec({geom}, {ba}, {dmap}, info);

    // order of stencil
    int linop_maxorder = 2;
    mlabec.setMaxOrder(linop_maxorder);
    
    // Set up boundary conditions (periodic for now)
    std::array<LinOpBCType,AMREX_SPACEDIM> bc_lo;
    std::array<LinOpBCType,AMREX_SPACEDIM> bc_hi;
    
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // lo-side BCs
        bc_lo[idim] = LinOpBCType::Periodic;
        // hi-side BCs
        bc_hi[idim] = LinOpBCType::Periodic;
    }

    // tell the solver what the domain boundary conditions are
    mlabec.setDomainBC(bc_lo, bc_hi);

    // set the boundary conditions if they are Dirichlet or Neumann
    // (does nothing for periodic case)
    mlabec.setLevelBC(0, &phi);

    // scaling factors
    Real ascalar = 0.0;
    Real bscalar = -1.0;
    mlabec.setScalars(ascalar, bscalar);

    // Set up coefficient matrices
    MultiFab acoef(ba, dmap, 1, 0);

    // fill in the acoef MultiFab and load this into the solver
    acoef.setVal(0.0);
    mlabec.setACoeffs(0, acoef);

    // bcoef lives on faces so we make an array of face-centered MultiFabs
    // then we will in face_bcoef MultiFabs and load them into the solver.
    std::array<MultiFab,AMREX_SPACEDIM> face_bcoef;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        const BoxArray& ba = amrex::convert(acoef.boxArray(),
                                            IntVect::TheDimensionVector(idim));
        face_bcoef[idim].define(ba, acoef.DistributionMap(), 1, 0);
        face_bcoef[idim].setVal(1.);
    }
    mlabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(face_bcoef));

    // build an MLMG solver
    MLMG mlmg(mlabec);

    // set solver parameters
    int max_iter = 100;
    mlmg.setMaxIter(max_iter);
    int max_fmg_iter = 0;
    mlmg.setMaxFmgIter(max_fmg_iter);
    int verbose = 2;
    mlmg.setVerbose(verbose);
    int bottom_verbose = 0;
    mlmg.setBottomVerbose(bottom_verbose);

    // relative and absolute tolerances for linear solve
    const Real tol_rel = 1.e-10;
    const Real tol_abs = 0.0;

    // Solve linear system
    mlmg.solve({&phi}, {&rhs}, tol_rel, tol_abs);

    MultiFab plotfile(ba, dmap, 2, 0);

    // copy data into plotfile MultiFab
    MultiFab::Copy(plotfile, rhs, 0, 0, 1, 0);
    MultiFab::Copy(plotfile, phi, 0, 1, 1, 0);

    // write out rhs and phi to plotfile
    WriteSingleLevelPlotfile("plt", plotfile, {"rhs","phi"}, geom, 1., 0);

    // Call the timer again and compute the maximum difference between the start time and stop time
    //   over all processors
    auto stop_time = amrex::second() - strt_time;
    const int IOProc = ParallelDescriptor::IOProcessorNumber();
    ParallelDescriptor::ReduceRealMax(stop_time,IOProc);

    // Tell the I/O Processor to write out the "run time"
    amrex::Print() << "Run time = " << stop_time << std::endl;
}
