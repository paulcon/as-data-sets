import numpy as np
from scipy.optimize import linprog

# checking to see if system has gurobi
try:
    HAS_GUROBI = True
    import gurobipy as gpy
except ImportError, e:
    HAS_GUROBI = False
    pass

# string constants for QP solver names
solver_SCIPY = 'SCIPY'
solver_GUROBI = 'GUROBI'

class QPSolver():
    """A class for solving linear and quadratic programs.

    Attributes
    ----------
    solver : str 
        identifies which linear program software to use

    Notes
    -----
    The class checks to see if Gurobi is present. If it is, it uses Gurobi to
    solve the linear and quadratic programs. Otherwise, it uses scipy
    implementations to solve the linear and quadratic programs.
    """
    solver = None

    def __init__(self, solver='GUROBI'):
        """Initialize a QPSolver.

        Parameters
        ----------
        solver : str, optional 
            identifies which linear program software to use. Options are 
            'GUROBI' and 'SCIPY'. (default 'GUROBI')
        """

        if solver==solver_GUROBI and HAS_GUROBI:
            self.solver = solver_GUROBI
        elif solver=='SCIPY':
            self.solver = solver_SCIPY
        else:
            self.solver = solver_SCIPY

    def general_linear_program(self, c, A_in, b_in, A_eq, b_eq, lb=None, ub=None):
        """Solves an inequality constrained linear program.

        This method returns the minimizer of the following linear program.

        minimize  c^T x
        A_eq x = b_eq
        subject to  A_in x >= b_in
        lb <= x <= ub

        Parameters
        ----------
        c : ndarray
            m-by-1 matrix for the linear objective function
        A_in : ndarray
            L-by-m matrix that contains the coefficients of the linear inequality 
            constraints
        b_in : ndarray 
            size L-by-1 matrix that is the right hand side of the inequality 
            constraints
        A_eq : ndarray
            M-by-m matrix that contains the coefficients of the linear equality 
            constraints
        b_eq : ndarray 
            size M-by-1 matrix that is the right hand side of the equality 
            constraints
        lb, ub : ndarray
            m-length arrays with upper and lower bounds on x, or None

        Returns
        -------
        x : ndarray
            m-by-1 matrix that is the minimizer of the linear program
        
        """

        if len(A_in.shape) == 1: A_in.reshape((1, A_in.shape[0]))
        if len(A_eq.shape) == 1: A_eq.reshape((1, A_eq.shape[0]))

        if self.solver == solver_SCIPY:
            return _scipy_general_program(c, A_in, b_in, A_eq, b_eq, lb, ub)
        elif self.solver == solver_GUROBI:
            return _gurobi_general_program(c, A_in, b_in, A_eq, b_eq, lb, ub)
        else:
            raise ValueError('QP solver {} not available'.format(self.solver))

def _scipy_general_program(c, A_in, b_in, A_eq, b_eq, lb=None, ub=None):

    c = c.reshape((c.size,))
    b_in = b_in.reshape((b_in.size,))
    b_eq = b_eq.reshape((b_eq.size,))
    if lb is not None:
        lb = lb.reshape((len(lb), 1))
        ub = ub.reshape((len(ub), 1))

    # make bounds
    bounds = []
    if lb is not None:
        for i in range(lb.size):
            bounds.append((lb[i,0], ub[i,0]))
    else: bounds = None

    res = linprog(c, -A_in, -b_in, A_eq, b_eq, bounds, options={"disp": False})
    if res.success:
        return res.x.reshape((c.size,1))
    else:
        return None

def _gurobi_general_program(c, A_in, b_in, A_eq, b_eq, lb=None, ub=None):
    
    b_in = b_in.reshape((len(b_in), 1))
    b_eq = b_eq.reshape((len(b_eq), 1))
    c = c.reshape((len(c), 1))
    if lb is not None:
        lb = lb.reshape((len(lb), 1))
        ub = ub.reshape((len(ub), 1))
    
    m,n = A_eq.shape
    model = gpy.Model()
    model.setParam('OutputFlag', 0)

    # Add variables to model
    vars = []
    if(lb is not None):
        for j in range(n):
            vars.append(model.addVar(lb=lb[j,0], ub=ub[j,0], vtype=gpy.GRB.CONTINUOUS))
    else:
        for j in range(n):
            vars.append(model.addVar(lb=0,
                    ub=gpy.GRB.INFINITY, vtype=gpy.GRB.CONTINUOUS))
    model.update()

    # Populate linear constraints
    for i in range(m):
        expr = gpy.LinExpr()
        for j in range(n):
            expr += A_eq[i,j]*vars[j]
        model.addConstr(expr, gpy.GRB.EQUAL, b_eq[i,0])

    m,n = A_in.shape

    for i in range(m):
        expr = gpy.LinExpr()
        for j in range(n):
            expr += A_in[i,j]*vars[j]
        model.addConstr(expr, gpy.GRB.GREATER_EQUAL, b_in[i,0])

    # Populate objective
    obj = gpy.LinExpr()
    for j in range(n):
        obj += c[j,0]*vars[j]
    model.setObjective(obj)
    model.update()

    # Solve
    model.optimize()

    if model.status == gpy.GRB.OPTIMAL:
        return np.array(model.getAttr('x', vars)).reshape((n,1))
    else:
        return None

class Polytope():
    
    def __init__(self, vertices, adjacency=True):
        """
        vertices is an Nxd array of N vertices in R^d.
        If adjacency is True, compute the adjacency matrix.
        """
        self.V = vertices
        if(adjacency): self.compute_adjacency()

    def compute_adjacency(self):
        """
        Compute the adjacency matrix.
        """
        self.adj = np.zeros((self.V.shape[0], self.V.shape[0]), dtype=int)
        for i in range(self.V.shape[0]):
            for j in range(self.V.shape[0]):
                if(j < i):
                    self.adj[i,j] = self.adj[j,i]
                elif(j > i):
                    self.adj[i,j] = check_adjacency(self.V, i, j)
        
def check_adjacency(V, i, j):
    """
    This checks the adjacency of vertex i and vertex j of a polytope.
    NOTE: THIS ISN'T THE CORRECT WAY TO DO THIS, BUT IT WORKS FOR ZONOTOPES;
    IF THE SUMMANDS HAVE MORE THAN 2 VERTICES, THIS MAY NOT WORK
    
    Parameters
    ----------
    V : the array of polytope vertices (row-wise)
    i, j : the indices of the vertices to test
    
    Returns
    -------
    True if V[i] is adjacent to V[j], False otherwise
    """
    tol = 1e-2
    solver = QPSolver()
    a = V[i]
    b = V[j]
    c = np.ones(len(a))
    A_in = np.zeros((1, V.shape[1]))
    for k in range(V.shape[0]):
        if k not in [i,j]: A_in = np.vstack((A_in, (a - V[k])/np.linalg.norm(a - V[k])))
    A_in = A_in[1:,]
    b_in = np.zeros(A_in.shape[0])
    A_eq = (a - b).reshape((1,len(a)))/np.linalg.norm(a - b)
    b_eq = np.array([0])
    lb = -1*np.ones(len(c))
    ub = np.ones(len(c))
    res = solver.general_linear_program(c, -A_in, b_in+tol, A_eq, b_eq, lb, ub)
    if res is not None: return True
    else: return False

def is_parallel(u, v):
    """
    This returns True if u and v (vectors) are parallel, False otherwise
    """
    if u is None or v is None: return False
    a = u/np.linalg.norm(u)
    b = v/np.linalg.norm(v)
    if np.all(np.isclose(a,b)) or np.all(np.isclose(-a,b)): return True
    else: return False

def compute_e_dict(decomps, v_ind, poly_list):
    """
    This computes e_j(v_j, i).
    
    Parameters
    ----------
    decomps : the set of decompositions of the vertices of the sum, row-wise
    v_ind : the index of the vertex we're examining
    poly_list : the list of polytopes being summed
    
    Returns
    -------
    e_dict : a dictionary with the tuples (j,i) as keys and the corresponding edge (or None) as values
    """
    Delta = [(j, i) for j in range(len(poly_list)) for i in range(poly_list[j].V.shape[0])]
    e_dict = {}
    for t in Delta:
        j = t[0]; i = t[1]
        if poly_list[j].adj[decomps[v_ind,j],i] == 1:
            e_dict[(j, i)] = poly_list[j].V[i] - poly_list[j].V[decomps[v_ind,j]]
        else: e_dict[(j, i)] = None
    return e_dict

def Adj(decomps, v_ind, poly_list):
    """
    This is the adjacency oracle.
    
    Parameters
    ----------
    decomps : the set of decompositions of the vertices of the sum, row-wise
    v_ind : the index of the vertex we're examining
    poly_list : the list of polytopes being summed

    Returns
    -------
    ret_dict : a dictionary containing the tuples (v_ind, (j,i)) as keys and the corresponding
        adjacent vertex (and its decomposition) as values
    """
    Delta = [(j, i) for j in range(len(poly_list)) for i in range(poly_list[j].V.shape[0])]
    e_dict = compute_e_dict(decomps, v_ind, poly_list)
    
    # this is called Delta(v, s, r) in the paper
    Delta_s_r = {}
    for t in Delta:
        z = t[0]; y = t[1]
        if e_dict[(z, y)] is not None:
            sr_list = []
            for t1 in Delta:
                j = t1[0]; i = t1[1]
                if is_parallel(e_dict[(j, i)], e_dict[(z, y)]): sr_list.append((j, i))
            Delta_s_r[(z, y)] = sr_list
        else: Delta_s_r[(z, y)] = None
        
    k = len(poly_list); d = poly_list[0].V.shape[1]; c = np.zeros(d+1);c[-1]=1; tol = -1e-15
    A_eq = np.zeros((1,d+1)); b_eq = np.array([0])
    ub = np.ones(d+1); lb = -ub
    
    ret_dict = {}
    S = QPSolver()
    for t in Delta:
        if e_dict[t] is not None:
            # linear feasibility problem determining whether e_s(v_s,r) is an edge of P
            A_in = (-e_dict[t]/np.linalg.norm(e_dict[t])).reshape((1, d))
            for t1 in Delta:
                if e_dict[t1] is not None and t1 not in Delta_s_r[t]: A_in = np.vstack((A_in, e_dict[t1]/np.linalg.norm(e_dict[t1])))
            A_in = np.hstack((A_in, np.zeros((A_in.shape[0], 1)))); A_in[0,-1] = 1
            b_in = np.zeros(A_in.shape[0])
            res = S.general_linear_program(c, A_in, b_in, A_eq, b_eq, lb, ub)

            # if it is an edge, find the adjacent vertex
            if res is not None and res[-1]<tol:
                dec = np.zeros((1, k))
                vhat = np.zeros((1, d))
                for j in range(k):
                    a_flag = False
                    for i in range(poly_list[j].V.shape[0]):
                        if (j,i) in Delta_s_r[t]:
                            vhat += poly_list[j].V[i]
                            dec[0,j] = i
                            a_flag = True
                            break
                    if not a_flag: 
                        vhat += poly_list[j].V[decomps[v_ind,j]]
                        dec[0,j] = decomps[v_ind,j]
                ret_dict[(v_ind,t)] = (vhat, dec)

            else: ret_dict[(v_ind,t)] = None
        else: ret_dict[(v_ind,t)] = None

    return ret_dict                        

def local_search(decomps, poly_list):
    """
    The local search function.

    Parameters
    ----------
    decomps : a 1xJ array containing the decomposition of the vertex we're examining (there are J summands)
    poly_list : the list of polytopes we are summing
    
    Returns
    -------
    A dictionary with decomps (as a tuple) for the key and the returned vertex (with its decomposition) for the value
    """
    Delta = [(j, i) for j in range(len(poly_list)) for i in range(poly_list[j].V.shape[0])]
    e_dict = compute_e_dict(decomps, 0, poly_list)
    
    # find a vector (c) in the normal cone of the vertex we're examining
    A_in = np.zeros((1, poly_list[0].V.shape[1]))
    for t in Delta:
        if e_dict[t] is not None: A_in = np.vstack((A_in, e_dict[t]/np.linalg.norm(e_dict[t])))
    b_in = np.zeros((A_in.shape[0],1)); b_in[0] = 1
    A_in = np.hstack((A_in, np.ones((A_in.shape[0],1))))
    c = np.zeros((A_in.shape[1], 1)); c[-1] = -1
    A_eq = np.zeros((1,A_in.shape[1])); b_eq = np.array([0])
    ub = np.ones(A_in.shape[1]); lb = -ub; lb[-1]=0
    c = QPSolver().general_linear_program(c, -A_in, -b_in, A_eq, b_eq, lb, ub)[:-1].T
    
    # cstar (normalized one-vector) is in the normal cone of vstar by design
    cstar = np.ones((1, poly_list[0].V.shape[1]))
    cstar = cstar/np.linalg.norm(cstar)

    # some items for linear programs
    S = QPSolver()
    cop = np.ones(1)

    # if c and cstar are parallel, we take c to be c + epsilon*c_perp, where
    # c_perp is perpendicular to c and epsilon is small enough that c + epsilon*c_perp
    # is still in the normal cone of v
    if is_parallel(c, cstar):
        # c_perp is perpendicular to c
        c_perp = np.ones(c.shape); c_perp[0,-1] = 1 - c.shape[1]
        # find an epsilon such that c + epsilon*c_perp is on v's normal cone's boundary
        for t in Delta:
            j = t[0]; i = t[1]
            dj = poly_list[j].V[decomps[0,j]].T
            vij = poly_list[j].V[i].T
            A_eq = (c_perp - c).dot(dj - vij)
            b_eq = -c.dot(dj - vij)
            A_in = np.zeros(1); b_in = np.zeros(1)
            for t1 in Delta:
                if t1 != t:
                    j1 = t1[0]; i1 = t1[1]
                    dj = poly_list[j1].V[decomps[0,j1]].T
                    vij = poly_list[j1].V[i1].T
                    A_in = np.vstack((A_in, (c_perp - c).dot(dj - vij)))
                    b_in = np.vstack((b_in, -c.dot(dj - vij)))
            epsilon = S.general_linear_program(cop, A_in.reshape((len(A_in), 1)), b_in, A_eq.reshape((len(A_eq), 1)), b_eq)
            if epsilon is not None and epsilon > 0: break
        # if such an epsilon cna't be found, set it to be somethiing small
        if epsilon is None: epsilon = 1e-10
        c = c + epsilon/2.0*c_perp
    c = c/np.linalg.norm(c)
    
    # find theta such that c + theta(cstar - c) is on v's normal cone's boundary
    theta = None
    lb = np.zeros(1); ub = np.ones(1)
    for t in Delta:
        j = t[0]; i = t[1]
        dj = poly_list[j].V[decomps[0,j]].T
        vij = poly_list[j].V[i].T
        A_eq = (cstar - c).dot(dj - vij)
        b_eq = -c.dot(dj - vij)
        A_in = np.zeros(1); b_in = np.zeros(1)
        for t1 in Delta:
            if t1 != t:
                j1 = t1[0]; i1 = t1[1]
                dj = poly_list[j1].V[decomps[0,j1]].T
                vij = poly_list[j1].V[i1].T
                A_in = np.vstack((A_in, (cstar - c).dot(dj - vij)))
                b_in = np.vstack((b_in, -c.dot(dj - vij)))
        theta = S.general_linear_program(cop, A_in.reshape((len(A_in), 1)), b_in, A_eq.reshape((len(A_eq), 1)), b_eq, lb, ub)
        if theta is not None and 0 < theta < 1: break
    if theta is None: theta = 0
    
    # find the vertex whose normal cone has the point c + theta(cstar - c)
    dec = np.zeros((1, len(poly_list)))
    vhat = np.zeros((1, poly_list[0].V.shape[1]))
    # theta has something small added to it for numeracy
    chat = (c + (theta+1e-10)*(cstar - c)).T
    for i in range(len(poly_list)):
        P = poly_list[i]
        j = np.argmax(P.V.dot(chat))
        vhat += P.V[j]
        dec[0,i] = j
    key = tuple([d for d in decomps[0]])
    return {key:(vhat, dec)}

def MinkSum(polytope_list):
    """
    This implements the reverse search algorithm for finding the Minkowski Sum of polytopes,
    described in the paper From the zonotope construction to the Minkowski addition of 
    convex polytopes (K. Fukuda, 2004).
    
    Parameters
    ----------
    polytope_list : a list of the polytopes to be summed
    
    Returns
    -------
    ret_vertices : a matrix where each row is a vertex of the Minkowski sum
    """

    # This finds the vertex whose normal cone contains the one-vector
    c = np.ones((polytope_list[0].V.shape[1], 1))
    vstar = np.zeros((1, polytope_list[0].V.shape[1]))
    vertex_mink_decomps = np.zeros((1, len(polytope_list)))
    for i in range(len(polytope_list)):
        P = polytope_list[i]
        j = np.argmax(P.V.dot(c))
        vstar += P.V[j]
        vertex_mink_decomps[0,i] = j
    ret_vertices = vstar
    
    # Adj_dict and local_dict are dictionaries containing the adjacency oracle and local search function results
    Adj_dict = {}
    local_dict = {}
    Delta = [(j, i) for j in range(len(polytope_list)) for i in range(polytope_list[j].V.shape[0])];Delta.reverse()
    end_t = Delta[0]; t = None
    v = vstar; v_ind = 0
    
    # the primary loop of the algorithm
    while not(np.all(v == vstar) and t == end_t):
        while t != end_t:
            # increment (j,i) by one
            t = Delta.pop()
            # get the next vertex according to the adjacency oracle
            try:
                next_v = Adj_dict[(v_ind,t)]
            except:
                Adj_dict.update(Adj(vertex_mink_decomps, v_ind, polytope_list))
                next_v = Adj_dict[(v_ind,t)]

            if next_v is not None:
                key = tuple([elem for elem in next_v[1][0]])
                # evaluate the local search function at the next vertex
                try:
                    f_next_v = local_dict[key]
                except:
                    local_dict.update(local_search(next_v[1], polytope_list))
                    f_next_v = local_dict[key]
                    
                # if f(next) = v
                if np.all(f_next_v[1] == vertex_mink_decomps[v_ind]):
                    # output v (update the ret_vertices and the decompositions) and reset Delta/t
                    v = next_v[0]
                    ret_vertices = np.vstack((ret_vertices, v))
                    vertex_mink_decomps = np.vstack((vertex_mink_decomps, next_v[1]))
                    # find the index corresponding to the new v
                    for k in range(vertex_mink_decomps.shape[0]):
                        if np.all(vertex_mink_decomps[k] == next_v[1]): v_ind = k
                    Delta = [(j, i) for j in range(len(polytope_list)) for i in range(polytope_list[j].V.shape[0])];Delta.reverse()
                    t = None
        
        # if(v /= v^*) then (* forward traverse *)
        if not np.all(vertex_mink_decomps[0] == vertex_mink_decomps[v_ind]):
            # u = v
            u_ind = v_ind
            
            # v = f(v)
            key = vertex_mink_decomps[v_ind].reshape((1,len(polytope_list)))
            key = tuple([elem for elem in key[0]])            
            try:
                f_v = local_dict[key]
            except:
                local_dict.update(local_search(vertex_mink_decomps[v_ind].reshape((1,len(polytope_list))), polytope_list))
                f_v = local_dict[key]                
            v = f_v[0]
            # find the corresponding index
            for k in range(vertex_mink_decomps.shape[0]):
                if np.all(vertex_mink_decomps[k] == f_v[1]): v_ind = k
                
            # reset Delta/t
            Delta = [(j, i) for j in range(len(polytope_list)) for i in range(polytope_list[j].V.shape[0])];Delta.reverse()
            t = None
            
            # restore (j,i) such that Adj(v,(j,i)) = u
            flag = False            
            while not flag:
                t = Delta.pop()
                try:
                    Adj_v_t = Adj_dict[(v_ind, t)]
                except:
                    Adj_dict.update(Adj(vertex_mink_decomps, v_ind, polytope_list))
                    Adj_v_t = Adj_dict[(v_ind, t)]
                if Adj_v_t is not None and np.all(Adj_v_t[1] == vertex_mink_decomps[u_ind]):
                    flag = True            
    
    return ret_vertices