# run_hallthruster.jl -- batch driver for the HallThruster.jl (v0.18.7) thruster component of the
# coupled SPT-100 model of Marks, Eckels, Mora & Gorodetsky (J. Appl. Phys., 2025).
#
# Usage: julia --project=<env with HallThruster v0.18.7> run_hallthruster.jl <bfield.csv> <in.csv> <out.csv>
#
# in.csv rows (comma separated, no header):
#   idx, Vd [V], mdot_a [kg/s], PB [Torr], Vcc [V], Tec [eV], alpha_anom, beta_anom, z_anom,
#   L_anom, dz_anom, u_n [m/s], c_w, f_n [, ncells, duration [s], avg_start [s]]
# The three optional held settings default to 100 cells, 1 ms and 0.5 ms (the data set's values).
# out.csv rows (appended, one per run, flushed after each):
#   idx, status (0 ok, 1 HallThruster retcode not success, 2 exception, 3 non-finite output),
#   ID [A], IB [A], T [N], u_exit [m/s], ID_std [A], T_std [N], ID_a [A], ID_b [A], T_a [N], T_b [N], wall_s
# ID_a/ID_b and T_a/T_b are averages over the first and second halves of the averaging window.
#
# Written for the as-data-sets project (MIT licence). Calls HallThruster.jl (MIT, UM-PEPL).

using HallThruster: HallThruster as het
using DelimitedFiles
using Printf

const LCH = 0.025          # SPT-100 channel length [m]
const DOMAIN = (0.0, 3 * LCH)  # three channel lengths (paper, Sec. II A 2)
const DURATION = 1.0e-3    # s (default)
const AVG_START = 0.5e-3   # average over the last 500 us (default)
const NUM_SAVE = 1000
const NCELLS = 100         # default

function make_thruster(bfile)
    data = readdlm(bfile, ',')
    geom = het.Geometry1D(; inner_radius = 0.035, outer_radius = 0.05, channel_length = LCH)
    bf = het.MagneticField(; file = "", z = Float64.(data[:, 1]), B = Float64.(data[:, 2]))
    return het.Thruster(; name = "SPT-100", geometry = geom, magnetic_field = bf, shielded = false)
end

function initial_dt(Vd, Vcc, ncells)
    # conservative CFL-0.2 starting step (adaptive stepping takes over)
    mi = 131.293e-3 / 6.02214076e23
    dx = DOMAIN[2] / (ncells + 1)
    u = sqrt(2 * 1.602176634e-19 * max(Vd - Vcc, 1.0) / mi)
    return 0.2 * dx / u
end

function run_one(thruster, row)
    Vd, mdot, PB, Vcc, Tec, alpha, beta, zan, Lan, dzan, un, cw, fn = row[1:13]
    ncells, duration, avg_start = length(row) >= 16 ? (Int(row[14]), row[15], row[16]) : (NCELLS, DURATION, AVG_START)
    anom = het.SimpleLogisticShift(;
        model = het.ScaledGaussianBohm(; anom_scale = alpha, barrier_scale = beta, center = zan, width = Lan),
        shift_length = dzan, midpoint_pressure = 25.0e-6, slope = 2.0,
    )
    config = het.Config(;
        thruster = thruster,
        domain = DOMAIN,
        discharge_voltage = Vd,
        anode_mass_flow_rate = mdot,
        propellant = het.Xenon,
        ncharge = 1,
        cathode_coupling_voltage = Vcc,
        cathode_Tev = Tec,
        anom_model = anom,
        wall_loss_model = het.WallSheath(het.BNSiO2, cw),
        ion_wall_losses = true,
        background_pressure_Torr = PB,
        neutral_ingestion_multiplier = fn,
        neutral_velocity = un,
        solve_plume = true,
        apply_thrust_divergence_correction = false,
    )
    sim = het.SimParams(;
        grid = het.EvenGrid(ncells), dt = initial_dt(Vd, Vcc, ncells), duration = duration,
        num_save = NUM_SAVE, verbose = false, print_errors = false, adaptive = true,
        min_dt = 1.0e-10, max_dt = 1.0e-7,
    )
    sol = het.run_simulation(config, sim)
    if sol.retcode != :success
        return 1, fill(NaN, 10)
    end
    i0 = findfirst(>=(avg_start), sol.t)
    n = length(sol.t)
    imid = i0 + (n - i0 + 1) ÷ 2
    Id = het.discharge_current(sol)[i0:end]
    Th = het.thrust(sol)[i0:end]
    avg = het.time_average(sol, i0)
    ID = het.discharge_current(avg, 1)
    IB = het.ion_current(avg, 1)
    T = het.thrust(avg, 1)
    z = avg.params.grid.cell_centers
    ui = avg.frames[1].ui[1, :]
    # linear interpolation of singly-charged ion velocity at the exit plane z = L_ch
    k = searchsortedlast(z, LCH)
    uexit = ui[k] + (ui[k + 1] - ui[k]) * (LCH - z[k]) / (z[k + 1] - z[k])
    m = imid - i0
    out = [ID, IB, T, uexit, std(Id), std(Th),
           sum(Id[1:m]) / m, sum(Id[(m + 1):end]) / (length(Id) - m),
           sum(Th[1:m]) / m, sum(Th[(m + 1):end]) / (length(Th) - m)]
    return all(isfinite, out) ? (0, out) : (3, out)
end

std(x) = sqrt(sum((x .- sum(x) / length(x)) .^ 2) / (length(x) - 1))

function main(args)
    bfile, infile, outfile = args
    thruster = make_thruster(bfile)
    rows = readdlm(infile, ',', Float64)
    done = Set{Int}()
    if isfile(outfile)
        for l in eachline(outfile)
            isempty(strip(l)) || push!(done, parse(Int, split(l, ',')[1]))
        end
    end
    open(outfile, "a") do io
        for r in 1:size(rows, 1)
            idx = Int(rows[r, 1])
            idx in done && continue
            t0 = time()
            status, out = try
                run_one(thruster, rows[r, 2:end])
            catch err
                msg = sprint(showerror, err)
                println(stderr, "run $idx exception: ", first(msg, 400))
                (2, fill(NaN, 10))
            end
            wall = time() - t0
            println(io, join([string(idx), string(status), [@sprintf("%.10e", v) for v in out]..., @sprintf("%.2f", wall)], ","))
            flush(io)
            @printf("run %d status %d ID %.4f T %.5f (%.1f s)\n", idx, status, out[1], out[3], wall)
            flush(stdout)
        end
    end
end

main(ARGS)
