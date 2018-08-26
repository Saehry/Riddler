'''
Program for solving the Riddler Classic puzzle found here:
https://fivethirtyeight.com/features/how-many-hoops-will-kids//
-jump-through-to-play-rock-paper-scissors/

'''
import numpy as np
import matplotlib.pyplot as plt

def get_transitions(N):
    # Gets the Markov transition matrix where the states are positions
    #(0,0),(0,1), ..., (0,N + 1), (1, 1), (1, 2),...(1,N)
    # The state (a, b) has someone from team A in hoop a, and someone from team
    # B in hoop B.  There are N physical hoops: "hoop 0" is the team A base, and
    # "hoop N + 1" is the team B base
    # Since we only have states where a <= b, we do some upper triangular math
    # to make the indices work out OK
    # We'll also never actually have states (0, 0) or (N + 1, N + 1), but the indexing
    # is easier if we leave them in, and two extra states shouldn't slow things down
    # too much.
    P = np.array([])
    for a in range(0, (N + 1) + 1):
        for b in range(a, (N + 1) + 1):
            probs = np.zeros((N + 2, N + 2))
            #If students haven't met yet, apprach each other
            if b - a > 1:
                probs[a + 1][b - 1] = 1.0
            #If game has ended, stay ended
            elif a == 0 and b == 1 or a == N and b == N + 1:
                probs[a][b] = 1.0
            #If student have met, assign probabilities
            elif b >= a and not (a == 0 and b == 0) and not (a == N + 1 and b == N + 1):
                probs[a][b] = 1.0/3.0     # Tie in rock, paper, scissors
                probs[0][b] = 1.0/3.0     # B wins
                probs[a][N + 1] = 1.0/3.0 # A wins
            # add probabilites to matrix (flattened, only caring about upper triangular states)
            if P.size == 0:
                P = probs[np.triu_indices(N + 2)]
            else:
                P = np.vstack((P, [probs[np.triu_indices(N + 2)]]))
    return np.transpose(P)

def take_step(P, N, state=np.array([])):
    # Given transition matrix P for N hoops and current state distribution, return
    # state distribution after one step.  Initializes distribution to the starting
    # position (0, N + 1) if not specified
    if state.size == 0:
        state = get_start(N)
    return np.matmul(P, state)

def get_start(N):
    # returns starting state distribution for N hoops
    start = np.zeros(((N + 2)*(N + 3)//2, 1))
    start[N + 1] = 1
    return start  #start in state (0, N + 1)

def expected_length(N, P=np.array([]), state=np.array([]), end_P = 0, end_E = 0, stop_P = 0.9999999):
    # For N hoops, determine the distribution of number of steps to finish
    # Can specify transition probabilities, initial distribution, initial
    # probability of finishing, and tolerance if desired, otherwise
    # gets transitions from N, begins at start state, and waits for
    # 0.99999999 of the probability of finishing to accumulate
    # Returns number of time-steps taken, and expected length of game in seconds

    #Initialize as necessary
    if P.size == 0:
        P = get_transitions(N)
    if state.size == 0:
        state = get_start(N)

    E = end_E
    t = 0
    while end_P < stop_P:
        t += 1
        state = take_step(P, N, state)
        # Expected number of steps to finish with max t steps is expected number of
        # steps to finish with max t - 1 steps plus
        # t*P[finishing in t steps|didn't finish in <= t - 1 steps]
        # Two end states are (0,1) (A wins, state[1]) and (N, N + 1) (B wins, state[-2])
        E += t*(state[1] + state[-2] - end_P)
        # Update probability we have finished
        end_P = state[1] + state[-2]
    return t, E[0]

def min_N_for_T(T, start_N = 0, show_intermediate=True):
    # Determines the minimum number of hoops to give a game with
    # expected length of at least T seconds.  Can start at a minimum
    # N if you know it takes at least that many hoops.
    # Shows expected times for intermediate N if desired
    # Returns vectors of N values and corresponding E values,
    # ending with the desireed values
    (t, E) = expected_length(start_N)
    if t > T:
        N = 0
    else:
        N = start_N
    N_vec = []
    E_vec = []
    while E < T:
        N += 1
        (t, E) = expected_length(N)
        N_vec = np.append(N_vec, N)
        E_vec = np.append(E_vec, E)
        if show_intermediate:
            nice_print(N, E, t)
    return (N_vec, E_vec)


def nice_print(N, E, t):
    # Prints a string telling you the expected gametime E for N hoops, and the
    # number of timesteps t taken to work this out
    print("N = {0:3} hoops, E = {1:2}:{2:04.3}, (t = {3:5})".format(N, int(E//60), E % 60, t))
    return

def fit_results_quadratic(N_vec, E_vec):
    # Gets a quadratic fit for the expected time as a function of N
    return np.polyfit(N_vec, E_vec, 2, full=True)

def print_fit(fit):
    # Printer for a quadratic fit
    poly = np.poly1d(fit[0])
    res = fit[1]
    print("Results have quadratic approximation\n {0}\nwith residuals {1}.".format(poly, res[0]))
    return

def plot_results(N_vec, E_vec, fit=None):
    # Plots the results, adding a fit if provided
    # The format string comes from https://stackoverflow.com/
    # questions/23149155/printing-the-equation-of-the-best-fit-line
    if fit:
        fit_poly = np.poly1d(fit[0])
        domain = np.linspace(min(N_vec), max(N_vec), 100)
        plt.plot(domain, fit_poly(domain), '-', \
                 label = 'y=${}$'.format(''.join(['{}x^{}'.format(('{:.2f}'.format(j) \
                            if j<0 else '+{:.2f}'.format(j)),(len(fit_poly.coef)-i-1)) \
                                                  for i,j in enumerate(fit_poly.coef)])))
    plt.plot(N_vec, E_vec, '.', label="Results")
    plt.xlabel("Number of hoops")
    plt.ylabel("Expected time (s)")
    plt.title("Expected length of game as a function of number of hoops")
    plt.legend()
    plt.show()
    return  

if __name__ == "__main__":
    print("Part 1")
    N = 8
    (t, E) = expected_length(N)
    nice_print(N, E, t)
    
    print("\n\nPart 2")
    T = 30*60  #period length in seconds
    (N_vec, E_vec) = min_N_for_T(T)

    print("\n\nExtension: fitting")
    fit = fit_results_quadratic(N_vec, E_vec)
    print_fit(fit)
    plot_results(N_vec, E_vec, fit)
