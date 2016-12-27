#requires flatten! and unflatten!
import ..flatten!
import ..unflatten!


function swarmoptimize{F}(mlmodel::MLAlgorithm{F}, input::Matrix, answers::Matrix, cost::Function)

  #return this as the stepping function.
  #generate a storage array.
  parameters = length(mlmodel)
  initial_array = Vector{F}(parameters)
  flatten!(initial_array, mlmodel)

  cf = (data_array) -> begin
    unflatten!(mlmodel, data_array)
    nn_results = mlmodel(input)
    cost(answers, nn_results)
  end

  init = (population, a, cf) -> begin
    for idx = 1:size(population,2)
      population[:, idx] = a .* randn(size(population, 1))
    end
    return true
  end

  general_swarmopt(initial_array, cf, initializer = init)
end


#particle swarm optimizer, taken from other code by I. Yonemoto.

function general_swarmopt{F}(a::Array{F,1}, cf::Function; reporter::Function = (c, count, a)->(), popsize = 1000,
	initializer = default_initializer!,
	termcond = (bestscore, iter) -> iter > 100,
	acceptfn = (testscore, bestscore) -> prod(testscore .< bestscore))

	#Swarm optimization function.
	#Prerequisites:
	#		a is a seeding, linear array for the process.  For a default system you should seed with a zeros array.
	#		cf is a cost function that takes an array that looks like the seeding array.
	#		parameters:
	#		reporter is a function that gets called to do reports; c should be the current cost and count should be count.
	#
	#		popsize is how big of a swarm we should use.  Defaults to 5x the vector dimension
	#		maxiter is how many iterations it should take before we quit
	#		velfactor is the initial multiplicand for the velocity.
	#		accelfactor is what factor we should accelerate our swarm by each timestep.
	#		drag is the drag applied to the velocities
	#		sthresh is the score threshold - if we are below this score then we stop.

	#TODO: Some of these parameters will be moved over to the header.
	velfactor = F(0)
	accelfactor = F(0.5)
	drag = F(0.9)

	#determine the length of the array
	l = length(a)

	#set up the population matrix (array of test vectors)
	population = zeros(F, l, popsize)

  initializer(population, a, cf) || return a

	#set up the velocity matrix (array of vector deltas)
	velocities = velfactor * randn(F,l, popsize)

	#set up the vector that stores the best score
	bestscore = cf(population[:,1])
	minidx = 1

	#set up an attractor vector.
	attractor = zeros(F, l)

	#iterate to move the swarm.
  iter = 0
	while (true)
		termcond(bestscore, iter) && break

		iter += 1
		#find scores for the entire population, store them.
		for idx = 1:popsize
			#test this particular population member.
			testscore = cf(population[:,idx])
			#check if testscore is categorically better than the best.
			if acceptfn(testscore, bestscore)
				bestscore = testscore
				minidx = idx
			end
		end

		attractor = population[:,minidx]

		#generate an acceleration matrix and apply it to the velocities
		velocities += (accelfactor) * (attractor * ones(F, 1, popsize) - population)
		velocities[:, minidx] = zeros(F, l)

		#update the swarm
		population += velocities

		#apply drag
		velocities *= drag

		reporter(bestscore, iter, attractor)
	end

	#output the last attractor
	attractor

end #function
