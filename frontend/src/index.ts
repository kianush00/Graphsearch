import cytoscape from "cytoscape";
import $ from 'jquery';

class MathUtils {
    /**
     * Gets the distance between two nodes in a graph.
     *
     * @param node1 - The first node in the graph.
     * @param node2 - The second node in the graph.
     *
     * @returns The distance between the two nodes, rounded to the nearest whole number.
     */
    public static getDistanceBetweenNodes(node1: GraphNode, node2: GraphNode): number {
        const pos1 = node1.position
        const pos2 = node2.position

        const dx = pos1.x - pos2.x
        const dy = pos1.y - pos2.y

        return parseFloat(this.calculateEuclideanDistance(dx, dy).toFixed(0))
    }

    /**
     * Calculates the Euclidean distance between two points in a 2D space.
     *
     * @param dx - The difference in the x-coordinates of the two points.
     * @param dy - The difference in the y-coordinates of the two points.
     *
     * @returns The Euclidean distance between the two points.
     */
    public static calculateEuclideanDistance(dx: number, dy: number): number {
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * A static method that generates a random angle in radians.
     *
     * @returns {number} A random angle between 0 (inclusive) and 2pi (exclusive).
     */
    public static getRandomAngle(): number {
        return this.getRandomFloat(0.0, 2 * Math.PI)
    }

    /**
     * A static method that calculates the position of a node in a circular layout based on an angle and distance.
     *
     * @param angle - The angle in radians from the center of the circle.
     * @param distance - The distance from the center of the circle.
     *
     * @returns {Position} - An object containing the x and y coordinates of the node's position.
     *
     * @example
     * const angle = Math.PI / 4;
     * const distance = 100;
     * const position = QueryTermService.getAngularPosition(angle, distance);
     * console.log(position); // Output: { x: 70.71, y: 70.71 }
     */
    public static getAngularPosition(angle: number, distance: number): Position {
        return {
            x: distance * Math.cos(angle),
            y: distance * Math.sin(angle),
        }
    }

    /**
     * Generates a random floating-point number within a specified range.
     *
     * @param min - The minimum value (inclusive) of the range.
     * @param max - The maximum value (exclusive) of the range.
     *
     * @returns A random floating-point number within the range [min, max).
     *
     * @remarks
     * This function uses the `window.crypto.getRandomValues` method to generate a random number.
     * It then scales the random number to fit within the specified range and returns the result.
     */
    public static getRandomFloat(min: number, max: number): number {
        const randomBuffer = new Uint32Array(1);
        window.crypto.getRandomValues(randomBuffer);
        const randomNumber = randomBuffer[0] / (0xFFFFFFFF + 1); // 0xFFFFFFFF is the max value for Uint32
        return randomNumber * (max - min) + min;
    }

    /**
     * A static method that generates a random position within a circular area.
     * The position is represented as an object with x and y coordinates.
     *
     * @returns {Position} - An object representing the random position within the circular area.
     * The object has properties x and y, representing the coordinates of the position.
     *
     * @example
     * const randomPosition = MathUtils.getRandomAngularPosition();
     * console.log(randomPosition); // Output: { x: 123.45, y: 67.89 }
     */
    public static getRandomAngularPosition(): Position {
        const randomDistance = this.getRandomFloat(0.0, 200.0)
        const randomAngle = this.getRandomAngle()
        return this.getAngularPosition(randomAngle, randomDistance)
    }

    /**
     * A static method that generates a random angular position with a given distance.
     *
     * @param distance - The distance from the origin for the position.
     * @returns {Position} - An object representing the position with x and y coordinates.
     *
     * @example
     * const randomPosition = MathUtils.getRandomAngularPositionWithDistance(100);
     * console.log(randomPosition); // Output: { x: 70.71, y: 70.71 }
     */
    public static getRandomAngularPositionWithDistance(distance: number): Position {
        const randomAngle = this.getRandomAngle()
        return this.getAngularPosition(randomAngle, distance)
    }
}


class TextUtils {
    /**
     * A static utility function to generate a random string of a specified length.
     *
     * @param chars - The length of the random string to be generated.
     * @returns {string} - A random string of the specified length.
     *
     * @example
     * ```typescript
     * const randomString = getRandomString(10);
     * console.log(randomString); // Output: "aRFd4fK2Qj"
     * ```
     */
    public static getRandomString(chars: number): string {
        const charsList = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        let result = ''
        for (let i = 0; i < chars; i++) {
            result += charsList.charAt(Math.floor(MathUtils.getRandomFloat(0.0, charsList.length)))
        }
        return result
    }
}


class ConversionUtils {
    private static readonly minMaxDistancesUserGraph: [number, number] = [45.0, 125.0]
    private static readonly minDistanceSentenceGraph: number = 40.0
    private static readonly hopMinValue: number = 1.0

    /**
     * Converts the number of hops to the corresponding distance in the graph.
     *
     * @param hops - The number of hops from the central node to the neighbour term.
     * @param hopMaxValue - The maximum number of hops allowed in the graph.
     * @param userGraphConversion - A boolean indicating whether the graph is a user graph.
     *
     * @returns {number} - The distance from the central node to the neighbour term.
     * The distance is calculated based on the normalized value of the hops,
     * using a linear transformation between the minimum and maximum distances.
     *
     * @remarks
     * This function assumes that the minimum and maximum distances are defined.
     * It also assumes that the hopMinValue is 0.
     * If the hopMaxValue is less than 2, the function returns 1.0.
     */
    public static convertHopsToDistance(hops: number, hopMaxValue: number, userGraphConversion: boolean, graphZoom: number): number {
        if (hopMaxValue < 2) return this.minMaxDistancesUserGraph[0];
        const minMaxDistances = userGraphConversion ? this.minMaxDistancesUserGraph : [this.minDistanceSentenceGraph, 78 / graphZoom];
        const normalizedValue = this.normalize(hops, this.hopMinValue, hopMaxValue, minMaxDistances[0], minMaxDistances[1])
        return normalizedValue
    }

    /**
     * Converts a given distance to hops, based on a normalized value within a specified range.
     *
     * @param distance - The distance to be converted to hops.
     * @param hopMaxValue - The maximum number of hops that can be achieved.
     * @param userGraphConversion - A boolean indicating whether the graph is a user graph.
     *
     * @returns {number} - The number of hops corresponding to the given distance, normalized within the specified range.
     *
     * @remarks
     * This function normalizes the given distance within the range of minimum and maximum distances,
     * and then maps the normalized value to the range of minimum and maximum hops.
     * If the hopMaxValue is less than 2, the function returns 1.0.
     * The returned value is rounded to one decimal place.
     */
    public static convertDistanceToHops(distance: number, hopMaxValue: number): number {
        if (hopMaxValue < 2) return 1.0
        const minMaxDistances =this.minMaxDistancesUserGraph;
        const normalizedValue = this.normalize(distance, minMaxDistances[0], minMaxDistances[1], this.hopMinValue, hopMaxValue)
        return parseFloat(normalizedValue.toFixed(1))
    }

    /**
     * Validates if the provided distance is out of the specified range.
     *
     * @param distance - The distance to be validated.
     *
     * @returns {boolean} - Returns `true` if the distance is out of range, `false` otherwise.
     *
     * @remarks
     * This function checks if the provided distance is less than the minimum distance or greater than the maximum distance.
     * If either condition is met, the function returns `true`, indicating that the distance is out of range.
     * Otherwise, it returns `false`, indicating that the distance is within the specified range.
     */
    public static validateDistanceOutOfRange(distance: number) : boolean {
        return distance < this.minMaxDistancesUserGraph[0] || distance > this.minMaxDistancesUserGraph[1]
    }

    /**
     * Normalizes a given value within a specified range.
     *
     * @param value - The value to be normalized.
     * @param oldMin - The minimum value of the original range.
     * @param oldMax - The maximum value of the original range.
     * @param newMin - The minimum value of the new range.
     * @param newMax - The maximum value of the new range.
     *
     * @returns The normalized value within the new range.
     * If the input value is less than oldMin or greater than oldMax, the function returns 1.0.
     *
     * @remarks
     * This function normalizes a given value by scaling it from the original range to the new range.
     * It uses a linear transformation formula to calculate the normalized value.
     */
    private static normalize(value: number, oldMin: number, oldMax: number, newMin: number, newMax: number): number {
        if (value < oldMin) {
            return newMin
        } else if (value > oldMax) {
            return newMax
        }

        const normalizedValue = newMin + ((value - oldMin) * (newMax - newMin)) / (oldMax - oldMin);
        return normalizedValue;
    }
}


class HTTPRequestUtils {
    /**
     * Sends a POST request to the specified endpoint with the provided data.
     *
     * @param endpoint - The endpoint to which the request will be sent.
     * @param data - The data to be sent in the request body.
     *
     * @returns A Promise that resolves with the response data.
     */
    public static async postData(endpoint: string, data: any): Promise<any> {
        try {
            const url = 'http://localhost:8080/'
            const response = await fetch(url + endpoint, {
                method: 'POST',
                headers: {
                'Content-Type': 'application/json'
                },
                body: JSON.stringify(data),
            })

            // Check the status code of the response
            const statusCode = response.status;
            console.log('HTTP Status Code:', statusCode);
        
            // Handle the response if code is successful
            const result = await response.json();
            if (response.ok) { // response.ok is true if code is between 200-299
                console.log('Success:', result);
                const sizeKb = new TextEncoder().encode(JSON.stringify(result)).length / 1024;
                console.log('Size of response in KB:', sizeKb);
                return result;
            } else {
                console.error('Error:', result);
                alert(`${result['detail']}`);
                return null;
            }
        } catch (error) {
            console.error('Error:', error)
            alert('An error occurred while sending the request.')
        }
    }

}



interface CyElementAggregator {
    addCyVisualElement(): void;
}

interface CyElementRemover {
    removeCyVisualElement(): void;
}

interface EdgeData {
    data: {
        id: string
        source: string
        target: string
        distance?: number
    }
}

class Edge implements CyElementAggregator, CyElementRemover {
    private readonly _id: string
    private readonly _sourceNode: GraphNode
    private readonly _targetNode: GraphNode
    private _distance: number
    private readonly _cyElement: cytoscape.Core

    /**
     * Represents an edge in the graph, connecting two nodes.
     * 
     * @param sourceNode - The source node of the edge.
     * @param targetNode - The target node of the edge.
     * @param isFromUserGraph - A boolean indicating whether the edge is from the user graph.
     */
    constructor(sourceNode: GraphNode, targetNode: GraphNode, isFromUserGraph: boolean) {
        this._id = "e_" + targetNode.id
        this._sourceNode = sourceNode
        this._targetNode = targetNode
        this._distance = MathUtils.getDistanceBetweenNodes(sourceNode, targetNode)
        this._cyElement = isFromUserGraph ? CytoscapeManager.getCyUserInstance() : CytoscapeManager.getCySentenceInstance();
        this.addCyVisualElement()
    }

    get distance(): number {
        return this._distance
    }

    /**
     * Sets the distance for the edge between the source and target nodes.
     * This method updates the distance value and the corresponding edge data in the graph.
     *
     * @param distance - The new distance for the edge.
     */
    set distance(distance: number) {
        this._distance = distance;
    }

    /**
     * Adds a visual edge to the graph interface.
     * 
     * This function creates a new edge in the graph using the `toObject` method,
     * which returns an object containing the edge's data.
     * The edge is then added to the graph using the `cy.add` method.
     */
    public addCyVisualElement(): void {
        this._cyElement.add(this.toObject())
    }

    /**
     * Removes the visual node from the graph interface.
     * 
     * This function removes the node with the given ID from the graph interface.
     * It uses the Cytoscape.js library to select the node by its ID and remove it from the graph.
     */
    public removeCyVisualElement(): void {
        this._cyElement.remove(this._cyElement.getElementById(this._id))
    }

    /**
     * Converts the Edge instance into a serializable object.
     * 
     * @returns An object containing the data of the Edge.
     * The object has properties: id, source and target.
     */
    public toObject(): EdgeData {
        const edgeData = {
            id: this._id,
            source: this._sourceNode.id,
            target: this._targetNode.id,
        };

        return { data: edgeData };
    }
}


type Position = {
    x: number
    y: number
}

enum NodeType {
    central_node,
    outer_node,
}

interface NodeData {
    data: {
        id: string,
        label: string,
    },
    position: Position
}

class GraphNode implements CyElementRemover {
    protected readonly _id: string
    protected _label!: string;
    protected _position: Position = {x: 0, y: 0};
    protected readonly _type: NodeType
    protected readonly _cyElement: cytoscape.Core
    
    /**
     * Represents a node in the graph.
     * 
     * @param id - The unique identifier of the node.
     * @param label - The label of the node.
     * @param position - The position of the node in the graph.
     * @param type - The type of the node.
     * @param isFromUserGraph - A boolean indicating whether the graph node belongs to the user graph.
     */
    constructor(id: string, label: string, type: NodeType, isFromUserGraph: boolean) {
        this._id = id
        this._type = type
        this._cyElement = isFromUserGraph ? CytoscapeManager.getCyUserInstance() : CytoscapeManager.getCySentenceInstance();
        this.label = label;
    }

    public get id(): string {
        return this._id;
    }

    public get position(): Position {
        return this._position;
    }

    public get label(): string {
        return this._label;
    }

    /**
     * Sets the label of the node and updates the associated graph node.
     *
     * @param label - The new label for the node.
     *
     * @remarks
     * This function updates the label of the node and also updates the label in the graph interface.
     * It uses the Cytoscape.js library to select the node by its ID and update the label data.
     */
    public set label(label: string) {
        this._label = label;
        this._cyElement.getElementById(this._id).data('label', label);
    }


    public setBackgroundColor(hexColor: string): void {
        this._cyElement.getElementById(this._id).style('background-color', hexColor);
    }

    /**
     * Removes the visual node from the graph interface.
     *
     * @remarks
     * This function removes the node with the given ID from the graph interface.
     * It uses the Cytoscape.js library to select the node by its ID and remove it from the graph.
     */
    public removeCyVisualElement(): void {
        this._cyElement.remove(this._cyElement.getElementById(this._id))
    }

    /**
     * Converts the GraphNode instance into a serializable object.
     *
     * @returns An object containing the data of the GraphNode.
     * The object has properties: id, label, and position.
     *
     * @remarks
     * This function is responsible for converting the GraphNode instance into a serializable object.
     * It returns an object containing the node's data and position.
     */
    public toObject(): NodeData {
        return {
            data: {
                id: this.id,
                label: this._label,
            },
            position: this._position,
        }
    }
}


class CentralNode extends GraphNode implements CyElementAggregator {
    /**
     * Represents a central node in the graph.
     * It manages the associated views, such as the visual node in the graph interface.
     * 
     * @param id - The unique identifier of the central node.
     * @param isUserGraphNode - A boolean indicating whether the central node belongs to the user graph.
     */
    constructor(id: string, isUserGraphNode: boolean) {
        const _id = id
        const label = id
        const type = NodeType.central_node
        super(_id, label, type, isUserGraphNode)
        this.addCyVisualElement()
    }

    /**
     * Adds a visual node to the graph interface.
     * 
     * This function creates a new node in the graph using the `toObject` method,
     * which returns an object containing the node's data and position.
     * The node is then added to the graph using the `cy.add` method.
     * The node's class is set to the string representation of the node's type.
     * The node is also locked and ungrabified to prevent user interaction.
     */
    public addCyVisualElement(): void {
        this._cyElement.add(this.toObject()).addClass(this._type.toString()).lock().ungrabify()
    }
}


class OuterNode extends GraphNode implements CyElementAggregator {
    /**
     * Constructor for the OuterNode class. Initializes a new OuterNode instance with the provided parameters.
     *
     * @param id - The unique identifier of the outer node.
     * @param label - The label of the outer node.
     * @param position - The position of the outer node in the graph.
     * @param isUserGraphNode - A boolean indicating whether the outer node belongs to the user graph.
     *
     * @remarks
     * The constructor sets the type of the outer node to `NodeType.outer_node`,
     * initializes the position of the outer node, and calls the `addCyVisualElement` method to add the node to the graph.
     */
    constructor(id: string, label: string, position: Position, isUserGraphNode: boolean) {
        const type = NodeType.outer_node
        super(id, label, type, isUserGraphNode)
        this.position = position;
        this.addCyVisualElement();
    }


    public get position(): Position {
        return this._position;
    }

    /**
     * Sets the position of the OuterNode in the graph.
     *
     * @param position - The new position for the OuterNode.
     *
     * @remarks
     * This function updates the position of the OuterNode and also updates the position in the graph interface.
     */
    public set position(position: Position) {
        this._position = position;
        this._updateVisualPosition();
    }

    /**
     * Adds a visual node to the graph interface.
     * 
     * @param isUserGraphNode - A boolean indicating whether the outer node belongs to the user graph.
     * 
     * @remarks
     * This function creates a new node in the graph using the `toObject` method,
     * which returns an object containing the node's data and position.
     * The node is then added to the graph using the `cy.add` method.
     * The node's class is set to the string representation of the node's type.
     */
    public addCyVisualElement(): void {
        const element = this._cyElement.add(this.toObject()).addClass(this._type.toString());
        if (this._cyElement === CytoscapeManager.getCySentenceInstance()) element.ungrabify();
    }

    /**
     * Sets the position of the OuterNode in the graph using an angle.
     *
     * @param angle - The angle from which to calculate the new position for the OuterNode.
     *
     * @remarks
     * This function calculates the new position of the OuterNode using the provided angle and the current distance from the central node.
     * It then updates the position of the OuterNode and the position in the graph interface.
     */
    public setPositionFromAngle(angle: number): void {
        this._position = MathUtils.getAngularPosition(angle, this._calculateDistance());
        this._updateVisualPosition();
    }

    /**
     * Sets the position of the OuterNode in the graph using a distance.
     *
     * @param distance - The distance from the central node to calculate the new position for the OuterNode.
     *
     * @remarks
     * This function calculates the new position of the OuterNode using a random angle and the provided distance.
     * It then updates the position of the OuterNode and the position in the graph interface.
     */
    public setPositionFromDistance(distance: number): void {
        this._position = MathUtils.getAngularPosition(MathUtils.getRandomAngle(), distance);
        this._updateVisualPosition();
    }

    /**
     * Calculates the distance from the central node to the OuterNode.
     *
     * @returns {number} - The distance from the central node to the OuterNode.
     *
     * @remarks
     * This function calculates the distance from the central node to the OuterNode using the Euclidean distance formula.
     */
    private _calculateDistance(): number {
        return MathUtils.calculateEuclideanDistance(this._position.x, this._position.y);
    }

    /**
     * Updates the visual position of the OuterNode in the graph interface.
     *
     * @remarks
     * This function is responsible for updating the position of the OuterNode in the graph interface.
     * It sets the position of the OuterNode to the current position of the OuterNode instance.
     */
    private _updateVisualPosition(): void {
        this._cyElement.getElementById(this._id).position(this._position)
    }
}


/**
 * Represents a term in a graph.
 * It is associated with a graph node.
 */
class Term {
    protected _value: string
    protected _node: GraphNode | undefined

    /**
     * Represents a term in a graph.
     * It is associated with a graph node.
     */
    constructor(value: string) {
        this._value = value
    }

    public get value(): string {
        return this._value;
    }

    public get node(): GraphNode | undefined {
        return this._node;
    }
}


interface ViewManager {
    displayViews(): void
    removeViews(): void
}

interface NTermObject {
    term: string;
    proximity_score: number;
    frequency_score: number;
    criteria: string;
}


type TermCriteria = "proximity" | "frequency" | "exclusion";

class NeighbourTerm extends Term implements ViewManager {
    protected declare _node: OuterNode | undefined
    private readonly _queryTerm: QueryTerm
    private _hops: number = 0.0
    private _nodePosition: Position = { x: 0, y: 0 }
    private _edge: Edge | undefined
    private readonly _proximityScore: number
    private readonly _frequencyScore: number
    private _criteria: TermCriteria;

    /**
     * Constructor for the NeighbourTerm class.
     * Initializes a new NeighbourTerm instance with the provided parameters.
     *
     * @param queryTerm - The QueryTerm instance associated with the neighbour term.
     * @param value - The value of the neighbour term.
     * @param proximityScore - The proximity score of the neighbour term.
     * @param frequencyScore - The frequency score of the neighbour term.
     * @param criteria - The criteria of the neighbour term, which can be 'proximity', 'frequency', or 'exclusion'.
     */
    constructor(queryTerm: QueryTerm, value: string, proximityScore: number, frequencyScore: number, 
        criteria: TermCriteria) {
        super(value)
        this._queryTerm = queryTerm
        this._proximityScore = proximityScore
        this._frequencyScore = frequencyScore
        this._criteria = criteria
        this._setInitialHops()
        this._value = value
    }

    public get proximityScore(): number {
        return this._proximityScore;
    }

    public get frequencyScore(): number {
        return this._frequencyScore;
    }

    public get criteria(): TermCriteria {
        return this._criteria;
    }
    
    /**
     * Displays the views of the query term and its associated neighbour terms in the graph.
     * This includes creating and positioning the CentralNode and OuterNodes.
     *
     * @returns {void} - This function does not return any value.
     *
     * @remarks
     * The function first checks if the CentralNode associated with the query term exists.
     * If the CentralNode does not exist, the function returns without performing any further actions.
     *
     * Next, the function creates a new OuterNode and sets its position and label.
     * It also creates a new Edge connecting the CentralNode and the OuterNode.
     *
     * Finally, the function sets the criteria for the term, which includes changing the node color based on the criteria.
     */
    public displayViews(): void {
        // Check if the CentralNode is defined in the graph
        const centralNode = this._queryTerm.node;
        if (centralNode === undefined) return 

        // Get the data for displaying the views
        const id = TextUtils.getRandomString(28);
        const isUserGraph = this._queryTerm.isUserGraph;

        // Build the outer node and its edge, and display them
        const outerNode = new OuterNode(id, this._value, this._nodePosition, isUserGraph);
        this._node = outerNode;
        this._edge = new Edge(centralNode, outerNode, isUserGraph);

        // Set the node color based on the criteria.
        this._initializeNodeColor();
    }

    /**
     * Removes the views of the neighbour terms and the central node.
     * 
     * This function is responsible for removing the visual nodes (OuterNodes and CentralNode)
     * and edges (connecting the CentralNode to the OuterNodes) from the graph interface.
     */
    public removeViews(): void {
        this._node?.removeCyVisualElement()
        this._edge?.removeCyVisualElement()
    }

    /**
     * Converts the NeighbourTerm instance into an object containing term, ponderation, and distance.
     * 
     * @returns {NTermObject} - An object with properties term, ponderation, and distance.
     * The term property contains the value of the NeighbourTerm instance.
     * The ponderation property contains the ponderation of the NeighbourTerm instance.
     * The distance property contains the number of hops from the central node to the NeighbourTerm instance.
     */
    public toObject(): NTermObject {
        const baseData = {
            term: this._value,
            proximity_score: this._proximityScore,
            frequency_score: this._frequencyScore,
            criteria: this._criteria
        };

        return baseData;
    }

    /**
     * Updates the position of the neighbour term node and updates the neighbour term's hops.
     *
     * @param neighbourTermsLength - The total number of neighbour terms associated with the query term.
     * @param index - The index of the neighbour term in the neighbour terms list.
     *
     * @returns {void} - This function does not return any value.
     * 
     * @remarks
     * This function calculates the new angle for the neighbour term node based on the index and the total number of neighbour terms.
     * It then calculates the distance from the central node using the neighbour term's hops.
     * The new position is obtained using the calculated angle and distance.
     * Finally, it updates the position of the neighbour term node and calls the `updateNodePosition` method.
     */
    public updateSymmetricalAngularPosition(neighbourTermsLength: number, index: number): void {
        const newAngle = (index / neighbourTermsLength) * Math.PI * 2 + 0.25
        const nodeDistance = ConversionUtils.convertHopsToDistance(this._hops, this._queryTerm.hopLimit, 
                this._queryTerm.isUserGraph, this._queryTerm.graphZoom);
        const nodePosition = MathUtils.getAngularPosition(newAngle, nodeDistance)
        this._updateNodePosition(nodePosition, nodeDistance)
    }

    /**
     * Sets the position of the neighbour term node and updates the neighbour term's hops.
     *
     * @param position - The new position of the neighbour term node.
     * The position object contains properties x and y representing the coordinates of the new position.
     *
     * @remarks
     * This function calculates the distance between the new position and the central node,
     * validates the position to ensure it falls within the specified range, updates the number of hops,
     * and updates the position of the neighbour term node.
     *
     * @returns {void} - This function does not return any value.
     */
    public updateNodePositionAndHops(position: Position): void {
        // Calculate the values
        const nodeDistance = this._edge?.distance ?? 0
        const nodePosition = this._validatePositionWithinRange(position, nodeDistance)
        const distance = MathUtils.calculateEuclideanDistance(nodePosition.x, nodePosition.y)
        const hops = ConversionUtils.convertDistanceToHops(distance, this._queryTerm.hopLimit)

        // Update the hops and node position
        this._updateHops(hops)
        this._updateNodePosition(nodePosition, distance)
    }

    /**
     * Updates the number of hops for the neighbour term.
     * If the query term belongs to the user graph, the function calculates the previous hops,
     * updates the current hops, and calls the `updateUserCriteria` method to update the criteria.
     *
     * @param newHops - The new number of hops from the central node to the neighbour term.
     *
     * @returns {void} - This function does not return any value.
     */
    private _updateHops(newHops: number): void {
        if (this._queryTerm.isUserGraph) {
            const previousHops = this._hops
            this._hops = newHops
            this._updateUserCriteria(previousHops, newHops)
        }
    }

    /**
     * Validates the position of a neighbour term node within a specified range.
     * If the position is outside the range, it adjusts the position to be within the range.
     *
     * @param position - The position of the neighbour term node to be validated.
     * @param nodeDistance - The distance of the neighbour term node from the central node.
     *
     * @returns {Position} - The validated position of the neighbour term node.
     *
     * @remarks
     * This function checks if the position of the neighbour term node is within the specified range (50.0 to 200.0 units from the central node).
     * If the position is outside the range, it adjusts the position to be within the range by calculating the adjusted X and Y coordinates.
     * The adjusted position is then returned.
     */
    private _validatePositionWithinRange(position: Position, nodeDistance: number): Position {
        const positionDistance = MathUtils.calculateEuclideanDistance(position.x, position.y)

        if (this._edge !== undefined && this._node !== undefined ) {
            if (ConversionUtils.validateDistanceOutOfRange(positionDistance)) {
                const angle = Math.atan2(position.y, position.x)
                const adjustedX = Math.cos(angle) * nodeDistance
                const adjustedY = Math.sin(angle) * nodeDistance
                position.x = adjustedX
                position.y = adjustedY
            }
        }
        return position
    }

    /**
     * Updates the position and distance of the node in the graph.
     *
     * @param nodePosition - The new position of the node in the graph.
     * @param distance - The new distance between the node and its connected neighbour term.
     *
     * @returns {void} - This function does not return any value.
     *
     * @remarks
     * This function updates the position and distance of the node in the graph.
     * It checks if the node and edge exist before updating their properties.
     */
    private _updateNodePosition(nodePosition: Position, distance: number): void {
        this._nodePosition = nodePosition;
        if (this._node !== undefined) this._node.position = this._nodePosition;
        if (this._edge !== undefined) this._edge.distance = distance;
    }

    /**
     * Sets the initial number of hops for the neighbour term based on the user graph status.
     *
     * @returns {void} - This function does not return any value.
     *
     * @remarks
     * If the query term belongs to the user graph, the initial hops are set to 1.0.
     * If the query term does not belong to the user graph, the initial hops are set to the query hop limit.
     * This initial hop value is used to calculate the position of the neighbour term in the graph.
     */
    private _setInitialHops(): void {
        this._hops = this._queryTerm.isUserGraph ? 1.0 : this._queryTerm.hopLimit;
    }

    /**
     * Initializes the color of the node based on the criteria.
     *
     * @remarks
     * This function checks the value of the `criteria` property and calls the appropriate method to set the node's color.
     * If the criteria is 'proximity', it calls `setProximityNodeColor()`.
     * If the criteria is 'frequency', it calls `setFrequencyNodeColor()`.
     * If the criteria is neither 'proximity' nor 'frequency', it calls `setExclusionNodeColor()`.
     *
     * @returns {void} - This function does not return any value.
     */
    private _initializeNodeColor(): void {
        if (this._criteria === "proximity") {
            this._setProximityNodeColor();
        } else if (this._criteria === "frequency") {
            this._setFrequencyNodeColor();
        } else {
            this._setExclusionNodeColor();
        }
    }

    /**
    * Updates the criteria of the neighbour term based on the number of hops.
    *
    * @param previousHops - The number of previous hops from the central node to the neighbour term.
    * @param newHops - The number of new hops from the central node to the neighbour term.
    *
    * @returns {void} - This function does not return any value.
    *
    * @remarks
    * This function checks the number of hops and updates the criteria of the neighbour term accordingly.
    * If the number of hops is less than 1.7, the criteria is set to "proximity".
    * If the number of hops is between 1.7 and 3.2 (exclusive), the criteria is set to "frequency".
    * If the number of hops is greater than or equal to 3.2, the criteria is set to "exclusion".
    */
    private _updateUserCriteria(previousHops: number, newHops: number): void {
        if (previousHops >= 1.7 && newHops < 1.7) {
            this._criteria = "proximity";
            this._setProximityNodeColor();
        } else if ((previousHops < 1.7 || previousHops >= 3.2) && (newHops >= 1.7 && newHops < 3.2)) {
            this._criteria = "frequency";
            this._setFrequencyNodeColor();
        } else if (previousHops < 3.2 && newHops >= 3.2) {
            this._criteria = "exclusion";
            this._setExclusionNodeColor();
        }
    }

    private _setProximityNodeColor(): void {
        this._node?.setBackgroundColor("#73b201");
    }

    private _setFrequencyNodeColor(): void {
        this._node?.setBackgroundColor("#2750db");
    }

    private _setExclusionNodeColor(): void {
        this._node?.setBackgroundColor("#FF0000");
    }
}


/**
 * Represents a query term that is associated with a central node in the graph.
 * It also manages neighbour terms related to the query term.
 */
class QueryTerm extends Term implements ViewManager {
    protected declare _node: CentralNode | undefined;
    private _neighbourTerms: NeighbourTerm[] = []
    private readonly _isUserGraph: boolean
    private _individualQueryTermsList: string[] = []
    private readonly _hopLimit: number
    private _graphZoom: number = 1.2

    /**
     * Constructor for the QueryTerm class.
     * Initializes a new QueryTerm instance with the provided value, user graph status, and hop limit.
     *
     * @param value - The value of the query term.
     * @param isUserGraph - A boolean indicating whether the query term belongs to the user graph.
     * @param hopLimit - The maximum number of hops allowed for neighbour terms in the document.
     */
    constructor(value: string, isUserGraph: boolean, hopLimit: number) {
        super(value);
        this._isUserGraph = isUserGraph;
        this._hopLimit = hopLimit
        this._updateGraphZoom();
    }

    public get isUserGraph(): boolean {
        return this._isUserGraph;
    }

    public get hopLimit(): number {
        return this._hopLimit;
    }

    public get individualQueryTermsList(): string[] {
        return this._individualQueryTermsList;
    }

    public set individualQueryTermsList(queryTermsList: string[]) {
        this._individualQueryTermsList = queryTermsList;
    }

    public get graphZoom(): number {
        return this._graphZoom;
    }

    public get neighbourTerms(): NeighbourTerm[] {
        return this._neighbourTerms;
    }

    public set neighbourTerms(neighbourTerms: NeighbourTerm[]) {
        this._neighbourTerms = neighbourTerms;
        this._updateGraphZoom();
        this._updateOuterNodesAngles();
    }

    /**
     * Displays the views of the query term and its associated neighbour terms in the graph.
     * This includes creating and positioning the CentralNode and OuterNodes.
     */
    public displayViews(): void {
        this._node = new CentralNode(this._value, this._isUserGraph)
        for (let neighbourTerm of this._neighbourTerms) {
            neighbourTerm.displayViews();
        }
        this._centerAndZoomNode();
    }

    /**
    * Removes the views of the neighbour terms and the central node.
    */
    public removeViews(): void {
        for (let neighbourTerm of this._neighbourTerms) {
            neighbourTerm.removeViews()
        }
        this._node?.removeCyVisualElement()
    }

    public getNeighbourTermsValues(): string[] {
        return this._neighbourTerms.map(term => term.value)
    }

    public getNeighbourProximityTermsValues(): string[] {
        return this._neighbourTerms
            .filter(term => term.criteria === "proximity")
            .map(term => term.value);
    }

    public getNeighbourFrequencyTermsValues(): string[] {
        return this._neighbourTerms
            .filter(term => term.criteria === "frequency")
            .map(term => term.value);
    }

    public getNeighbourTermsAsObjects(): NTermObject[] {
        return this._neighbourTerms.map(term => term.toObject())
    }

    public getNeighbourTermByNodeId(id: string): NeighbourTerm | undefined {
        return this._neighbourTerms.find(nterm => nterm.node?.id === id)
    }

    public getNeighbourTermByValue(value: string): NeighbourTerm | undefined {
        return this._neighbourTerms.find(nterm => nterm.value === value)
    }

    public addNeighbourTerm(neighbourTerm: NeighbourTerm): void {
        this._neighbourTerms.push(neighbourTerm)
        this._updateGraphZoom();
        this._updateOuterNodesAngles()
    }

    public removeNeighbourTerm(neighbourTerm: NeighbourTerm): void {
        this._neighbourTerms = this._neighbourTerms.filter(term => term !== neighbourTerm)
        neighbourTerm.removeViews()
        this._updateGraphZoom();
        this._updateOuterNodesAngles()
    }

    private _updateOuterNodesAngles(): void {
        for (let i = 0; i < this._neighbourTerms.length; i++) {
            this._neighbourTerms[i].updateSymmetricalAngularPosition(this._neighbourTerms.length, i)
        }
    }

    private _updateGraphZoom(): void {
        this._graphZoom = this._isUserGraph ? 1.2 : this._getPersonalizedZoomIfSentenceGraph();
    }

    /**
     * Centers the graph on the CentralNode.
     * 
     * This function is responsible for zooming in the graph and centering it on the CentralNode.
     * It first zooms in the graph by a factor of 1.2, then checks if the visible query term has a node.
     * If the node exists and is a CentralNode, it centers the graph on the node.
     */
    private _centerAndZoomNode(): void {
        const cyElement = this._isUserGraph ? CytoscapeManager.getCyUserInstance() : CytoscapeManager.getCySentenceInstance();

        // Zoom the graph. If its a sentence graph, then do a personalized zoom based on the lenght of neighbour terms
        cyElement.zoom(this._graphZoom);

        // Center the graph on the CentralNode, if it exists
        if (this._node === undefined) return;
        cyElement.center(cyElement.getElementById(this._node.id))

        // Pan the graph vertically, if it's a sentence graph, to make it easier to see the neighbour terms
        if (!this._isUserGraph) cyElement.panBy({ x: 0, y: -1 });
    }

    /**
     * Calculates and returns the personalized zoom level for the sentence graph based on the number of neighbour terms.
     *
     * @returns {number} - The personalized zoom level for the sentence graph.
     * The zoom level is calculated based on the number of neighbour terms as follows:
     * - If the number of neighbour terms is less than or equal to 15, the zoom level is 1.2.
     * - If the number of neighbour terms is between 16 and 60 (inclusive), the zoom level is calculated by subtracting
     *   a fraction from 1.2, where the fraction is proportional to the difference between the number of neighbour terms and 15.
     * - If the number of neighbour terms is greater than 60, the zoom level is calculated by dividing 0.2 by
     *   a factor that is proportional to the difference between the number of neighbour terms and 60.
     */
    private _getPersonalizedZoomIfSentenceGraph(): number {
        if (this.neighbourTerms.length <= 15) {
            return 1.2;
        } else if (this.neighbourTerms.length <= 60) {
            return 1.2 - ((this.neighbourTerms.length - 15) / 45);
        } else {
            return 0.2 / (1 + 0.01 * (this.neighbourTerms.length - 60));
        }
    }
}


class TextElement {
    protected _queryTerm: QueryTerm

    /**
    ​ * Constructor for the QueryTerm class.
    ​ * Initializes a new QueryTerm instance with the provided query term value and neighbour terms.
    ​ *
    ​ * @param queryTermValue - The value of the query term.
    ​ * @param responseNeighbourTerms - An array of objects representing neighbour terms retrieved from the response.
    ​ * Each object should have properties: term, distance, and ponderation.
    ​ * @param hopLimit - The maximum number of hops allowed for the neighbour terms in the document.
    ​ */
    constructor(queryTermValue: string, responseNeighbourTerms: any[], hopLimit: number) {
        this._queryTerm = new QueryTerm(queryTermValue, false, hopLimit)
        this._initializeNeighbourTermsFromResponse(responseNeighbourTerms)
    }

    get queryTerm(): QueryTerm {
        return this._queryTerm
    }

    /**
     * Initializes neighbour terms from the response data.
     * 
     * @param responseNeighbourTerms - An array of objects containing neighbour term data retrieved from the response.
     * Each object has properties: term, distance, and ponderation.
     * 
     * @returns {void} - This function does not return any value.
     */
    private _initializeNeighbourTermsFromResponse(responseNeighbourTerms: any[]): void {
        const neighbourTerms = []
        for (const termObject of responseNeighbourTerms) {
            const neighbourTerm = new NeighbourTerm(this._queryTerm, termObject.term, termObject.proximity_score, 
                termObject.frequency_score, termObject.criteria)
            neighbourTerms.push(neighbourTerm)
        }
        this._queryTerm.neighbourTerms = neighbourTerms;
    }
}



interface SentenceObject {
    position_in_doc: number;
    raw_text: string;
    all_neighbour_terms: NTermObject[];
}

// Format: [ [first_idx, last_idx, raw_word, processed_word], ... ]
type RawToProcessedMap = [number, number, string, string][];

class Sentence extends TextElement {
    private readonly _positionInDoc: number
    private readonly _rawText: string
    private readonly _rawToProcessedMap: RawToProcessedMap

    /**
    ​ * Constructor for the Sentence class.
    ​ * Initializes a new Sentence instance with the provided query term value, neighbour terms, and sentence details.
    ​ *
    ​ * @param queryTermValue - The value of the query term associated with the sentence.
    ​ * @param responseNeighbourTerms - An array of objects representing neighbour terms retrieved from the response.
    ​ * Each object should have properties: term, distance, and ponderation.
    ​ * @param hopLimit - The maximum number of hops allowed for the neighbour terms in the document.
    ​ * @param positionInDoc - The position of the sentence in the document.
    ​ * @param rawText - The raw text of the sentence.
     * @param rawToProcessedMap - The map of raw to processed words in the sentence
    ​ */
    constructor(queryTermValue: string, responseNeighbourTerms: any[], hopLimit: number, positionInDoc: number, 
        rawText: string, rawToProcessedMap: RawToProcessedMap){
        super(queryTermValue, responseNeighbourTerms, hopLimit)
        this._positionInDoc = positionInDoc
        this._rawText = rawText
        this._rawToProcessedMap = rawToProcessedMap
    }

    get rawText(): string {
        return this._rawText
    }

    get rawToProcessedMap(): RawToProcessedMap {
        return this._rawToProcessedMap
    }

    public toObject(): SentenceObject {
        return {
            position_in_doc: this._positionInDoc,
            raw_text: this._rawText,
            all_neighbour_terms: this._queryTerm.getNeighbourTermsAsObjects()
        }
    }
}


interface DocumentObject {
    doc_id: string;
    title: string;
    abstract: string;
    preprocessed_text: string;
    weight: number;
    all_neighbour_terms: NTermObject[];
}

class Document extends TextElement {
    private readonly _id: string
    private readonly _title: string
    private readonly _abstract: string
    private readonly _preprocessed_text: string
    private readonly _weight: number
    private _isExcluded: boolean
    private readonly _sentences: Sentence[] = []

    /**
    ​ * Constructor for the Document class.
    ​ * Initializes a new Document instance with the provided query term value, neighbour terms, document details, and sentence data.
    ​ *
    ​ * @param queryTermValue - The value of the query term associated with the document.
    ​ * @param responseNeighbourTerms - An array of objects representing neighbour terms retrieved from the response.
    ​ * Each object has properties: term, distance, and ponderation.
    ​ * @param hopLimit - The maximum number of hops allowed for the neighbour terms in the document.
    ​ * @param idTitleAbstractPreprcsdtext - An array containing the document's id, title, abstract and preprocessed text.
    ​ * @param weight - The weight of the document.
    ​ * @param responseSentences - An array of objects representing sentences retrieved from the response.
    ​ * Each object has properties: position_in_doc, raw_text, and neighbour_terms.
    ​ */
    constructor(queryTermValue: string, responseNeighbourTerms: any[], hopLimit: number, idTitleAbstractPreprcsdtext: [string, string, string, string], 
        weight: number, responseSentences: any[]){
        super(queryTermValue, responseNeighbourTerms, hopLimit)
        this._id = idTitleAbstractPreprcsdtext[0]
        this._title = idTitleAbstractPreprcsdtext[1]
        this._abstract = idTitleAbstractPreprcsdtext[2]
        this._preprocessed_text = idTitleAbstractPreprcsdtext[3]
        this._weight = weight
        this._isExcluded = false
        this._sentences = this._initializeSentencesFromResponse(responseSentences, hopLimit)
    }

    get id(): string {
        return this._id
    }

    get title(): string {
        return this._title
    }

    get abstract(): string {
        return this._abstract
    }

    get sentences(): Sentence[] {
        return this._sentences
    }

    get isExcluded(): boolean {
        return this._isExcluded
    }

    set isExcluded(excluded: boolean) {
        this._isExcluded = excluded
    }

    public toObject(): DocumentObject {
        return {
            doc_id: this._id,
            title: this._title,
            abstract: this._abstract,
            preprocessed_text: this._preprocessed_text,
            weight: this._weight,
            all_neighbour_terms: this._queryTerm.getNeighbourTermsAsObjects()
        }
    }

    /**
    ​ * Initializes sentences from the provided response data.
    ​ * 
    ​ * @param responseSentences - An array of objects representing sentences retrieved from the response.
    ​ * Each object has properties 'position_in_doc', 'raw_text', and 'neighbour_terms'.
    ​ * 
    ​ * @param hopLimit - The maximum number of hops allowed for the neighbour terms in the sentences.
    ​ * 
    ​ * @returns An array of Sentence instances, each representing a sentence from the response data.
    ​ * Each Sentence instance is created with the query term value, neighbour terms, hop limit, position, and raw text.
    ​ */
    private _initializeSentencesFromResponse(responseSentences: any[], hopLimit: number): Sentence[] {
        const sentences = []
        for (const sentenceObject of responseSentences) {
            const sentence = new Sentence(this._queryTerm.value, sentenceObject.all_neighbour_terms, 
                    hopLimit, sentenceObject.position_in_doc, sentenceObject.raw_text, sentenceObject.raw_to_processed_map)
            sentences.push(sentence)
        }
        return sentences
    }
}


interface RankingObject {
    visible_neighbour_terms: NTermObject[];
    documents: DocumentObject[];
}

class Ranking {
    private readonly _visibleQueryTerm: QueryTerm
    private readonly _completeQueryTerm: QueryTerm
    private _documents: Document[] = []
    private _visibleSentence: Sentence | undefined

    /**
     * Initializes a new Ranking instance with the provided query term value and limit distance.
     *
     * @param queryTermValue - The value of the query term for which the Ranking instance will be created.
     * @param limitDistance - The maximum distance limit (hop limit) for neighbour terms.
     *
     * @returns A new Ranking instance with the provided query term value and limit distance.
     * The visible and complete QueryTerm instances are created with the provided query term value,
     * and the visibility and exclusion status of the documents are initialized to false.
     */
    constructor(queryTermValue: string, limitDistance: number) {
        this._visibleQueryTerm = new QueryTerm(queryTermValue, true, limitDistance)
        this._completeQueryTerm = new QueryTerm(queryTermValue, false, limitDistance)
    }


    get visibleQueryTerm(): QueryTerm {
        return this._visibleQueryTerm
    }

    get completeQueryTerm(): QueryTerm {
        return this._completeQueryTerm
    }

    get documents(): Document[] {
        return this._documents
    }

    get visibleSentence(): Sentence | undefined {
        return this._visibleSentence
    }

    /**
     * Sets the visible sentence in the ranking.
     * Removes the views of the current visible sentence (if any),
     * sets the new visible sentence, and displays the views of the new visible sentence.
     *
     * @param sentence - The new visible sentence to be set in the ranking.
     *
     * @returns {void} - This function does not return any value.
     */
    set visibleSentence(sentence: Sentence) {
        this._visibleSentence?.queryTerm.removeViews()
        this._visibleSentence = sentence
        this._visibleSentence.queryTerm.displayViews()
    }

    public getExcludedDocuments(): Document[] {
        return this._documents.filter(doc => doc.isExcluded)
    }

    public getNotExcludedDocuments(): Document[] {
        return this._documents.filter(doc => !doc.isExcluded)
    }

    public addDocument(document: Document): void {
        this._documents.push(document)
    }

    /**
     * Converts the Ranking instance into a plain JavaScript object (RankingObject).
     * 
     * @remarks
     * The RankingObject includes two properties:
     * - visible_neighbour_terms: An array of objects representing the neighbour terms associated with the visible QueryTerm.
     * - documents: An array of objects representing the documents associated with the Ranking.
     * 
     * @returns {RankingObject} - A plain JavaScript object containing the necessary data for the Ranking instance.
     */
    public toObject(): RankingObject {
        return {
            visible_neighbour_terms: this._visibleQueryTerm.getNeighbourTermsAsObjects(),
            documents: this._documents.map(document => document.toObject())
        }
    }

    /**
     * Updates the order of the documents in the ranking based on the provided excludedDocuments and newPositions arrays.
     *
     * @param excludedDocuments - An array of integers representing the indices of the documents to be excluded.
     * @param newPositions - An array of integers representing the new order of the documents.
     * Each integer corresponds to the index of a document in the documents array.
     *
     * @returns {void} - This function does not return any value.
     * It refreshes the exclusion status of all documents, sets the indices of the documents to be excluded,
     * and reorders the documents in the ranking based on the provided positions array.
     */
    public updateDocumentsOrder(excludedDocuments: number[], newPositions: number[]): void {
        this._refreshDocumentsExclusion();
        this._setExcludedDocuments(excludedDocuments);
        this._reorderDocuments(newPositions);
    }


    /**
     * Refreshes the exclusion status of all documents in the ranking.
     * This function iterates over the documents in the ranking and sets their exclusion status to false.
     *
     * @returns {void} - This function does not return any value.
     */
    private _refreshDocumentsExclusion(): void {
        this._documents.forEach(doc => doc.isExcluded = false);
    }

    /**
     * Sets the indices of the documents to be excluded in the ranking.
     * This function iterates over the provided excludedDocuments array,
     * checks if a document exists at each index, and sets its excluded property to true.
     * If a document does not exist at an index, logs a warning message to the console.
     *
     * @param excludedDocuments - An array of integers representing the indices of the documents to be excluded.
     *
     * @returns {void} - This function does not return any value.
     */
    private _setExcludedDocuments(excludedDocuments: number[]): void {
        for (const docIndex of excludedDocuments) {
            if (this._documents[docIndex] !== undefined) {
                this._documents[docIndex].isExcluded = true;
            } else {
                console.log(`Warning: Document at index ${docIndex} does not exist.`);
            }
        }
    }

    /**
     * Reorders the documents in the ranking based on the provided positions array.
     * If the lengths of the positions array and the documents array do not match,
     * logs an error message to the console and returns without modifying the documents.
     *
     * @param newPositions - An array of integers representing the new order of the documents.
     * Each integer corresponds to the index of a document in the documents array.
     *
     * @remarks
     * This function iterates over the positions array and creates a new array of documents
     * in the new order specified by the positions array.
     * It then assigns the reorderedDocuments array back to the documents property of the Ranking instance.
     */
    private _reorderDocuments(newPositions: number[]): void {
        const excludedDocuments: Document[] = this.getExcludedDocuments();
        const notExcludedDocuments: Document[] = this.getNotExcludedDocuments();

        // Validate the positions array length (it must be equal to the length of the not excluded documents array)
        if (newPositions.length !== notExcludedDocuments.length) {
            console.log('Warning: Positions array length must match documents array length.');
            alert('Something went wrong with the "Rerank" function.');
            return
        }

        // Reorder the not excluded documents
        const reorderedDocuments: Document[] = new Array(this._documents.length);
        for (let i = 0; i < newPositions.length; i++) {
            reorderedDocuments[i] = notExcludedDocuments[newPositions[i]];
        }

        // Add the excluded documents to the last positions
        let j = 0;
        for (let i = newPositions.length; i < this._documents.length; i++) {
            reorderedDocuments[i] = excludedDocuments[j];
            j++;
        }

        this._documents = reorderedDocuments;
    }
}


/**
 * A service class responsible for managing query terms and their associated data.
 */
class QueryTermService {
    private readonly _queryService: QueryService
    private readonly _ranking: Ranking
    private _isVisible: boolean = false

    /**
     * Initializes a new instance of the QueryTermService class.
     * 
     * @param queryService - The QueryService instance to be associated with the QueryTermService.
     * @param queryTermValue - The value of the query term for which the QueryTermService will be created.
     * @param limitDistance - The maximum distance limit for neighbour terms.
     */
    constructor(queryService: QueryService, queryTermValue: string, limitDistance: number) {
        this._queryService = queryService
        this._ranking = new Ranking(queryTermValue, limitDistance)
    }

    get visibleQueryTerm(): QueryTerm {
        return this._ranking.visibleQueryTerm;
    }

    get completeQueryTerm(): QueryTerm {
        return this._ranking.completeQueryTerm;
    }

    get ranking(): Ranking {
        return this._ranking
    }

    get visibleSentence(): Sentence | undefined {
        return this._ranking.visibleSentence;
    }

    set visibleSentence(sentence: Sentence) {
        this._ranking.visibleSentence = sentence;
    }

    /**
     * Retrieves data related to neighbour terms for the current query term.
     * The retrieved data is then used to create new NeighbourTerm instances,
     * which are added to the QueryTerm's neighbour terms list.
     * 
     * @param searchResults - The number of search results to retrieve.
     * @param limitDistance - The maximum distance limit for neighbour terms.
     * @param graphTerms - The number of neighbour terms to include in the graph.
     *
     * @returns {Promise<void>} - A promise that resolves when the data retrieval and processing are complete.
     */
    public async initialize(searchResults: number, limitDistance: number, graphTerms: number): Promise<void> {
        // Define the endpoint for retrieving neighbour terms data
        const endpoint = 'get-ranking';
        const query = this.visibleQueryTerm.value;
        const data = {query: query, search_results: searchResults, 
                limit_distance: limitDistance, graph_terms: graphTerms}

        try {
            // Send a POST request to the endpoint with the query term value
            const result = await HTTPRequestUtils.postData(endpoint, data)

            // Check if the result is not null
            if (result) {
                this.visibleQueryTerm.individualQueryTermsList = result['individual_query_terms_list'];
                this._generateVisibleNeighbourTerms(result)
                this._generateCompleteNeighbourTerms(result)
                this._generateRankingDocuments(result)
            } else {
                console.log("Warning: null API response");
            }
        } catch (error) {
            alert("Error retrieving data:" + error);
            console.error("Error retrieving neighbour terms data:", error)
        }
    }

    /**
     * If the node is dragged, updates the position of the neighbour term node and 
     * updates the neighbour term's hops.
     * @param id - The id of the neighbour term node.
     * @param position - The new position of the neighbour term node.
     */
    public nodeDragged(id: string, position: Position): void {
        // Check if the neighbour term exists
        const neighbourTerm = this.visibleQueryTerm.getNeighbourTermByNodeId(id)
        if (neighbourTerm === undefined) return;

        // Update the position of the neighbour term node and its hops
        neighbourTerm.updateNodePositionAndHops(position)

        // Update the neighbour terms table with the new hops values
        this._queryService.updateNeighbourTermsTable()
    }

    /**
     * Displays the QueryTerm and its associated views in the graph.
     * This includes creating and positioning the CentralNode and OuterNodes.
     */
    public display(): void {
        // Mark the QueryTerm as visible
        this._isVisible = true

        // Remove any existing views associated with the visible QueryTerm
        this.visibleQueryTerm.removeViews()

        // Display the views associated with the visible QueryTerm
        this.visibleQueryTerm.displayViews()
    }

    /**
     * This method removes the visual nodes and edges from the graph interface.
     */
    public deactivate(): void {
        // Mark the QueryTerm as not visible
        this._isVisible = false

        // Remove any existing views associated with the QueryTerm
        this.visibleQueryTerm.removeViews();
        this.visibleSentence?.queryTerm.removeViews();

        // Update the active term service in the QueryService's elements
        queryService.updateActiveTermServiceInElements(undefined);
    }

    /**
     * Adds a neighbour term to the QueryTerm's neighbour terms list.
     * It also updates the neighbour terms table in the QueryService.
     * If the QueryTerm is currently visible, it displays the views of the neighbour term.
     *
     * @param neighbourTerm - The neighbour term to be added.
     */
    public addVisibleNeighbourTerm(neighbourTerm: NeighbourTerm): void {
        // Ensure the visible query term has less than 20 neighbour terms
        if (this.visibleQueryTerm.neighbourTerms.length > 19) return;

        // Add the neighbour term to the visible query term's neighbour terms list
        this.visibleQueryTerm.addNeighbourTerm(neighbourTerm)

        // Update the neighbour terms table with the new neighbour term
        this._queryService.updateNeighbourTermsTable()
        this._queryService.updateAddTermsTable()

        // Display the views of the neighbour term if it is currently visible
        if (this._isVisible) this.display()
    }

    /**
     * Removes a neighbour term from the visible query term.
     * 
     * This function retrieves the neighbour term associated with the provided id,
     * and if found, removes it from the visible query term's neighbour terms list.
     * It then updates the neighbour terms table in the query service.
     * 
     * @param id - The id of the neighbour term to be removed.
     */
    public removeVisibleNeighbourTerm(id: string): void {
        // Retrieve the neighbour term associated with the provided id
        const neighbourTerm = this.visibleQueryTerm.getNeighbourTermByNodeId(id)

        // If the neighbour term is found and there are more than 1 neighbour term left, remove it
        if (neighbourTerm === undefined || this.visibleQueryTerm.neighbourTerms.length < 2) return;
        this.visibleQueryTerm.removeNeighbourTerm(neighbourTerm)

        // Update the neighbour terms table in the query service
        this._queryService.updateNeighbourTermsTable()
        this._queryService.updateAddTermsTable()

        // Display the views of the neighbour term if it is currently visible
        if (this._isVisible) this.display()
    }

    /**
     * Adds a neighbour term to the complete query term.
     * 
     * This function takes a NeighbourTerm instance as a parameter and adds it to the complete query term's neighbour terms list.
     * It also updates the add terms table in the QueryService.
     * 
     * @param neighbourTerm - The neighbour term to be added.
     * 
     * @returns {void} - This function does not return any value.
     */
    public addCompleteNeighbourTerm(neighbourTerm: NeighbourTerm): void {
        this.completeQueryTerm.addNeighbourTerm(neighbourTerm)
    }

    /**
    * Changes the cursor type of the HTML document to the specified newCursorType.
    * This function is used to update the cursor style when hovering over a neighbour term node in the graph.
    * 
    * @param id - The unique identifier of the neighbour term node.
    * @param newCursorType - The new cursor type to be applied to the HTML document.
    * 
    * @returns {void} - This function does not return any value.
    */
    public changeCursorType(id: string, newCursorType: string): void {
        // Check if the neighbour term is already in the graph
        const neighbourTerm = this.visibleQueryTerm.getNeighbourTermByNodeId(id)
        if (neighbourTerm === undefined) return

        // Update the cursor style for the HTML document
        $('html,body').css('cursor', newCursorType);
    }

    /**
     * Generates visible neighbour terms for the current query term.
     * 
     * @param result - The result object containing neighbour terms data.
     * The result object is expected to have a property 'visible_neighbour_terms',
     * which is an array of objects representing neighbour terms.
     * Each object should have properties 'term', 'distance', and 'ponderation'.
     * 
     * @returns {void} - This function does not return any value.
     */
    private _generateVisibleNeighbourTerms(result: any): void {
        // Iterate over the neighbour terms in the result
        for (let termObject of result['visible_neighbour_terms']) {
            // Create a new NeighbourTerm instance for each term object
            const neighbourTerm = this._initializeNewNeighbourTerm(this.visibleQueryTerm, termObject);

            // Add the neighbour term to the visible QueryTerm's neighbour terms list
            this.addVisibleNeighbourTerm(neighbourTerm)
        }
    }

    /**
     * Generates complete neighbour terms for the current query term.
     * 
     * @param result - The result object containing neighbour terms data.
     * The result object is expected to have a property 'complete_neighbour_terms',
     * which is an array of objects representing neighbour terms.
     * Each object should have properties 'term', 'distance', and 'ponderation'.
     * 
     * @returns {void} - This function does not return any value.
     * It iterates over the neighbour terms in the result, creates a new NeighbourTerm instance for each term object,
     * and adds the neighbour term to the complete QueryTerm's neighbour terms list.
     */
    private _generateCompleteNeighbourTerms(result: any): void {
        // Iterate over the neighbour terms in the result
        for (let termObject of result['complete_neighbour_terms']) {
            // Create a new NeighbourTerm instance for each term object
            const neighbourTerm = this._initializeNewNeighbourTerm(this.completeQueryTerm, termObject)

            // Add the neighbour term to the complete QueryTerm's neighbour terms list
            this.addCompleteNeighbourTerm(neighbourTerm)
        }

        // Update the neighbour terms table in the QueryService
        this._queryService.updateAddTermsTable()
    }

    /**
     * Initializes a new NeighbourTerm instance based on the provided term object and hop limit.
     * 
     * @param termObject - An object containing properties 'term', 'distance', and 'ponderation' representing a neighbour term.
     * 
     * @returns A new NeighbourTerm instance with the provided term value, distance, ponderation, and hop limit.
     */
    private _initializeNewNeighbourTerm(queryTerm: QueryTerm, termObject: any): NeighbourTerm {
        return new NeighbourTerm(queryTerm, termObject.term, termObject.proximity_score, 
            termObject.frequency_score, termObject.criteria)
    }

    /**
     * Generates ranking documents for the current query term, and updates the results list component.
     * 
     * @param result - The result object containing ranking documents data.
     * The result object is expected to have a property 'documents',
     * which is an array of objects representing documents.
     * Each object should have properties 'doc_id', 'title', 'abstract', and 'neighbour_terms'.
     * 
     * @returns {void} - This function does not return any value.
     */
    private _generateRankingDocuments(result: any): void {
        // Iterate over the documents in the result
        for (let documentObject of result['documents']) {
            const doc_id = documentObject['doc_id']
            const title = documentObject['title']
            const abstract = documentObject['abstract']
            const preprocessed_text = documentObject['preprocessed_text']
            const weight = documentObject['weight']
            const response_neighbour_terms = documentObject['all_neighbour_terms']
            const sentences = documentObject['sentences']
            const queryTermValue = this.visibleQueryTerm.value
            const hopLimit = this.visibleQueryTerm.hopLimit
            const document = new Document(queryTermValue, response_neighbour_terms, hopLimit, 
                    [doc_id, title, abstract, preprocessed_text], weight, sentences)
            this._addDocument(document)
        }

        // Update the ranking's documents list
        this._queryService.updateResultsList()
    }
    
    /**
     * Adds a document to the ranking and updates the results list.
     * 
     * @param document - The document to be added to the ranking.
     * 
     * @remarks
     * This function is responsible for adding a new document to the ranking and updating the results list.
     * It calls the `addDocument` method of the ranking
     */
    private _addDocument(document: Document): void {
        this.ranking.addDocument(document)
    }
}


class QueryService {
    private _activeQueryTermService: QueryTermService | undefined
    private readonly _queryTermServices: QueryTermService[]
    private readonly _neighbourTermsTable: NeighbourTermsTable
    private readonly _addTermsTable: AddTermsTable
    private readonly _queryTermsList: QueryTermsList
    private readonly _resultsList: ResultsList

    /**
     * Initializes a new instance of the QueryService class.
     * 
     * The QueryService class manages the query terms, neighbour terms, and results for the application.
     * It initializes the necessary components such as the neighbour terms table, results list, query terms list, and add terms table.
     * 
     * @remarks
     * The QueryService class is responsible for coordinating the interaction between different components of the application.
     */
    constructor() {
        this._queryTermServices = []
        this._neighbourTermsTable = new NeighbourTermsTable()
        this._resultsList = new ResultsList()
        this._queryTermsList = new QueryTermsList(this)
        this._addTermsTable = new AddTermsTable()
    }

    get activeQueryTermService(): QueryTermService | undefined { 
        return this._activeQueryTermService 
    }

    /**
     * Sets the query for the service.
     * Deactivates the currently active QueryTermService, creates a new Query object,
     * and triggers the query generation process.
     * 
     * @param queryValue - The new query string.
     * @param searchResults - The number of search results to retrieve.
     * @param limitDistance - The maximum distance limit for neighbour terms.
     * @param graphTerms - The number of neighbour terms to include in the graph.
     */
    public async setQuery(queryValue: string, searchResults: number, limitDistance: number, graphTerms: number): Promise<void> {
        // Deactivate the currently active service
        this._activeQueryTermService?.deactivate();

        // Wait for the new QueryTermService to be generated
        await this._generateNewQueryTermService(queryValue, searchResults, limitDistance, graphTerms);

        // Set the new service as active
        this.setActiveQueryTermService(queryValue);
    }

    /**
     * Sets the reranking for the service.
     * Sends a POST request to the 'rerank' endpoint with the provided ranking object.
     * If the response is successful, updates the ranking's documents exclusion, excluded documents,
     * and reorders the documents based on the received new positions.
     *
     * @param ranking - The ranking object containing the new positions and excluded documents.
     *
     * @returns {Promise<void>} - A promise that resolves when the reranking process is complete.
     */
    public async setRerank(ranking: RankingObject): Promise<void> {
        // Check if the active QueryTermService is defined
        if (this._activeQueryTermService === undefined) return;

        // Send the POST request
        const endpoint = 'rerank';

        try {
            const response = await HTTPRequestUtils.postData(endpoint, ranking);

            if (response) {
                // Handle the response accordingly
                const ranking_new_positions: number[] = response['ranking_new_positions'];
                const ranking_excluded_documents: number[] = response['ranking_excluded_documents'];

                // Update the ranking's documents exclusion, excluded documents, and reorder the documents
                this._activeQueryTermService.ranking.updateDocumentsOrder(ranking_excluded_documents, ranking_new_positions);

                // Finally, update the visual results list
                this.updateResultsList();
            } else {
                console.log("Warning: null API response");
            }
        } catch (error) {
            console.error("Error while operating rerank function:", error)
        }
    }

    /**
     * Sets the active QueryTermService based on the provided query value.
     * Deactivates the currently active QueryTermService, finds the corresponding QueryTermService,
     * by the provided queryValue, and displays the views associated with the QueryTerm.
     *
     * @param queryValue - The value of the query term for which to set the active QueryTermService.
     */
    public setActiveQueryTermService(queryValue: string): void {
        // Deactivate the active QueryTermService
        this._activeQueryTermService?.deactivate();

        // Set the QueryTermService as active
        const queryTermService = this._findQueryTermService(queryValue);
        if (queryTermService === undefined) return;
        this._activeQueryTermService = queryTermService

        // Display the views associated with the active QueryTermService
        this._activeQueryTermService.display()

        // Update the active term service in the QueryService's elements
        this.updateActiveTermServiceInElements(this._activeQueryTermService);
        
    }

    /**
     * Updates the active QueryTermService in the neighbour terms table, add terms table, and results list.
     * Also updates the corresponding tables and lists values in the UI.
     * 
     * @param queryTermService - The QueryTermService to be set as the active service.
     * If this parameter is `undefined`, the active service in each component will be deactivated.
     * 
     * @remarks
     * This function is responsible for coordinating the interaction between different components of the application.
     * It updates the active service in the neighbour terms table, add terms table, and results list.
     * If the provided `queryTermService` is `undefined`, it deactivates the active service in each component.
     * 
     * @returns {void}
     */
    public updateActiveTermServiceInElements(queryTermService: QueryTermService | undefined): void {
        this._neighbourTermsTable.activeTermService = queryTermService;
        this._addTermsTable.activeTermService = queryTermService
        this._resultsList.activeTermService = queryTermService;
    }


    public updateNeighbourTermsTable(): void {
        this._neighbourTermsTable.updateTable()
    }

    public updateResultsList(): void {
        this._resultsList.updateList()
    }

    public updateAddTermsTable(): void {
        this._addTermsTable.updateTable()
    }

    /**
     * Generates a new QueryTermService for a given query value.
     * This method checks if a QueryTermService for the given query value already exists.
     * If not, it creates a new QueryTermService, adds it to the queryTermServices array,
     * and updates the query terms list.
     *
     * @param queryValue - The value of the query term for which to generate a new QueryTermService.
     * @param searchResults - The number of search results to retrieve.
     * @param limitDistance - The maximum distance limit for neighbour terms.
     * @param graphTerms - The number of neighbour terms to include in the graph.
     */
    private async _generateNewQueryTermService(queryValue: string, searchResults: number, limitDistance: number, 
        graphTerms: number): Promise<void> {
        // Check if a QueryTermService for the given query value already exists
        if (this._findQueryTermService(queryValue) !== undefined) return;

        // Create a new QueryTermService and initialize it with the given parameters
        const queryTermService = new QueryTermService(this, queryValue, limitDistance);
        await queryTermService.initialize(searchResults, limitDistance, graphTerms);

        // Add the new QueryTermService to the queryTermServices array and update the query terms list
        this._queryTermServices.push(queryTermService)
        this._updateQueryTermsList()
        
    }

    /**
     * Finds and returns the QueryTermService associated with a given query value.
     * 
     * @param queryValue - The value of the query term for which to find the QueryTermService.
     * 
     * @returns The QueryTermService associated with the given query value, or undefined if no such service exists.
     * 
     * @remarks
     * This function iterates over the array of QueryTermServices and uses the Array.prototype.find method to find the service
     * with a visible query term value that matches the given query value.
     */
    private _findQueryTermService(queryValue: string): QueryTermService | undefined {
        return this._queryTermServices.find(
            termService => termService.visibleQueryTerm.value === queryValue
        )
    }

    /**
     * Updates the list of query terms from the interface, with new query terms.
     * 
     * This function retrieves the visible query terms from each QueryTermService,
     * and passes them to the QueryTermsList's updateList method to update the list.
     * 
     * @remarks
     * This function is responsible for keeping the list of query terms up-to-date
     * with the active QueryTermServices.
     * 
     * @returns {void}
     */
    private _updateQueryTermsList(): void {
        this._queryTermsList.updateList(
            this._queryTermServices.map(termService => termService.visibleQueryTerm)
        )
    }

}


class QueryTermsList {
    private readonly _dynamicList: HTMLElement
    private readonly _queryService: QueryService

    /**
     * Constructs a new instance of QueryTermsList.
     * 
     * @param queryService - The QueryService associated with the QueryTermsList.
     * 
     * @remarks
     * This constructor initializes the QueryTermsList by setting the queryService property and
     * retrieving the dynamicList element from the HTML structure.
     */
    constructor(queryService: QueryService) {
        this._queryService = queryService
        this._dynamicList = document.getElementById('queryTermsList') as HTMLElement
    }


    /**
     * Updates the list of query terms from the interface, with new query terms.
     * 
     * This function retrieves the visible query terms from each QueryTermService,
     * and passes them to the QueryTermsList's updateList method to update the list.
     * 
     * @param queryTerms - An array of QueryTerm objects representing the new query terms to be added to the list.
     * 
     * @returns {void}
     */
    public updateList(queryTerms: QueryTerm[]): void {
        // Clear existing list items
        this._dynamicList.innerHTML = ''

        // Iterate over the query terms and create list items for each one
        queryTerms.forEach(queryTerm => {
            // Create a new list item element
            const listItem = document.createElement("li")

            // Set the text content of the list item to be the value of the query term
            listItem.textContent = queryTerm.value

            // Add a click event listener to the list item
            listItem.addEventListener("click", () => {
                // When the list item is clicked, set the active query term service to the value of the query term
                this._queryService.setActiveQueryTermService(queryTerm.value)
            })

            // Append the list item to the dynamic list container
            this._dynamicList.appendChild(listItem)
        })
    }

}


class AddTermsTable {
    private _activeTermService: QueryTermService | undefined
    private readonly _dynamicTable: HTMLElement

    /**
     * Constructs a new instance of AddTermsTable.
     * 
     * Initializes the AddTermsTable by setting the dynamicTable property and adding an event listener to the filter input.
     * It also calls the toggleFilterVisibility method to handle the visibility of the filter input.
     */
    constructor() {
        this._dynamicTable = document.getElementById('addTermsTable') as HTMLElement
        const filterInput = document.getElementById('addTermsFilter') as HTMLInputElement;
        filterInput.addEventListener('input', () => this._filterTerms());
        this._toggleFilterVisibility();
    }

    /**
     * Sets the active QueryTermService and updates the table.
     *
     * @param queryTermService - The QueryTermService to be set as the active service.
     * This service will be used to retrieve and display data in the table.
     *
     * @returns {void} - This function does not return any value.
     *
     * @remarks
     * This function is responsible for setting the active QueryTermService and updating the table.
     * It is called whenever a new QueryTermService is selected by the user.
     */
    set activeTermService(queryTermService: QueryTermService | undefined) {
        this._activeTermService = queryTermService
        this.updateTable()
    }
    
    /**
    ​ * Updates the table with neighbour terms data.
    ​ * 
    ​ * This function retrieves the table body element, clears existing rows, and then iterates over the neighbour terms of the active query term.
    ​ * For each neighbour term, it creates a new row in the table and populates the cells with the term's value and hops.
    ​ * If the term is already in the visible neighbour terms list, it is not added to the table.
    ​ * 
    ​ * @remarks
    ​ * This function assumes that the table body element is already present in the HTML structure.
    ​ * 
    ​ * @returns {void}
    ​ */
    public updateTable(): void {
        // Get the table body element
        const tbody = this._dynamicTable.getElementsByTagName('tbody')[0]

        // Clear existing rows in the table
        tbody.innerHTML = '' 

        // Check if the activeTermService is defined
        if (this._activeTermService === undefined) return

        const visibleNeighbourTermsValues = this._activeTermService.visibleQueryTerm.getNeighbourTermsValues()

        // Iterate over the neighbour terms of the active query term
        for(const term of this._activeTermService.completeQueryTerm.neighbourTerms) {
            // Check if the term is not already in the visible neighbour terms list
            if ((!visibleNeighbourTermsValues.includes(term.value)) && (term.criteria === "proximity")) {
                // Create a new row in the table
                const row = tbody.insertRow()

                // Create cells for the row
                const cell1 = row.insertCell(0)
                const cell2 = row.insertCell(1)

                // Set the text content of the first cell
                cell1.innerHTML = term.value
                
                // Create the <i> element
                const icon = this._createIconElement(term)

                // Append the <i> element to the second cell
                cell2.appendChild(icon);
            }
        }

        // Toggle the filter input visibility, if the table has rows 
        this._toggleFilterVisibility()
    }

    /**
    ​ * Handles the addition of a neighbour term to the active query term's visible neighbour terms.
    ​ * 
    ​ * This function checks if the activeTermService is defined, retrieves the neighbour term by its value,
    ​ * and if the neighbour term exists, creates a new visible neighbour term and adds it to the active query term.
    ​ * 
    ​ * @param termValue - The value of the neighbour term to be added.
    ​ * 
    ​ * @returns {void}
    ​ */
    private _handleTermAddition(termValue: string): void {
        if (this._activeTermService === undefined) return;

        const neighbourTerm = this._activeTermService.completeQueryTerm.getNeighbourTermByValue(termValue)
        if (neighbourTerm === undefined) return;

        // Add the neighbour term to the active query term's visible neighbour terms
        const queryTerm = this._activeTermService.visibleQueryTerm;
        const value = neighbourTerm.value;
        const proximityScore = neighbourTerm.proximityScore;
        const frequencyScore = neighbourTerm.frequencyScore;
        const criteria = neighbourTerm.criteria;

        const visibleNeighbourTerm = new NeighbourTerm(queryTerm, value, proximityScore, frequencyScore, criteria);
        this._activeTermService.addVisibleNeighbourTerm(visibleNeighbourTerm);
    }

    /**
    ​ * Filters the terms in the 'addTermsTable' based on the input value in the 'addTermsFilter' input field.
    ​ * 
    ​ * This function retrieves the filter input element, the filter value, the table element, and the rows of the table.
    ​ * It then iterates over each row, retrieves the term cell, and checks if the term's lowercase value contains the filter value.
    ​ * If it does, the row's display style is set to '', making it visible. If it doesn't, the row's display style is set to 'none', making it hidden.
    ​ */
    private _filterTerms(): void {
        const filterInput = document.getElementById('addTermsFilter') as HTMLInputElement;
        const filterValue = filterInput.value.toLowerCase();
        const table = document.getElementById('addTermsTable') as HTMLTableElement;
        const rows = Array.from(table.getElementsByTagName('tbody')[0].getElementsByTagName('tr'));
    
        for (const row of rows) {
            const termCell = row.getElementsByTagName('td')[0];
            const term = termCell.textContent ?? termCell.innerText;
    
            // Check if the term contains the filter value and set the row's display style accordingly
            if (term.toLowerCase().indexOf(filterValue) > -1) {
                row.style.display = '';
            } else {
                row.style.display = 'none';
            }
        }
    }

    /**
     * Toggles the visibility of the addTermsFilter input based on the presence of rows in the addTermsTable.
     * 
     * This method checks whether there are any rows in the table's tbody. If there are no rows, the filter input
     * is hidden. If there are rows, the filter input is shown.
     */
    private _toggleFilterVisibility(): void {
        // Get the table and its tbody element
        const table = document.getElementById('addTermsTable') as HTMLTableElement;
        const tbody = table.getElementsByTagName('tbody')[0];

        // Get the filter input
        const filterInput = document.getElementById('addTermsFilter') as HTMLInputElement;

        // Check if there are any rows in the tbody
        const rows = tbody.getElementsByTagName('tr');
        
        if (rows.length === 0) {
            // Hide the filter input if there are no rows
            filterInput.style.display = 'none';
        } else {
            // Show the filter input if there are rows
            filterInput.style.display = 'block';
        }
    }

    /**
    ​ * Creates an icon element for adding a neighbour term to the active query term's visible neighbour terms.
    ​ * 
    ​ * @param term - The neighbour term for which to create the icon element.
    ​ * 
    ​ * @returns A new HTML element representing the icon.
    ​ * 
    ​ * The icon is a fontawesome plus-circle icon with a pointer cursor.
    ​ * When clicked, it triggers the `handleTermAddition` method with the term's value as a parameter.
    ​ */
    private _createIconElement(term: NeighbourTerm): HTMLElement {
        const icon = document.createElement('i');
        icon.className = 'fas fa-plus-circle';
        icon.style.cursor = 'pointer';

        // Add event listener to the icon element
        icon.addEventListener('click', () => {
            this._handleTermAddition(term.value);
        });

        return icon;
    }
}


class NeighbourTermsTable {
    private _activeTermService: QueryTermService | undefined
    private readonly _dynamicTable: HTMLElement

    /**
     * Constructor for the NeighbourTermsTable class.
     * Initializes the dynamicTable property with the HTML element with the id 'neighboursTermsTable'.
     */
    constructor() {
        this._dynamicTable = document.getElementById('neighboursTermsTable') as HTMLElement
    }

    /**
     * Sets the active QueryTermService and updates the table.
     *
     * @param queryTermService - The QueryTermService to be set as the active service.
     * This service will be used to retrieve and display data in the table.
     *
     * @returns {void} - This function does not return any value.
     *
     * @remarks
     * This function is responsible for setting the active QueryTermService and updating the table.
     * It is called whenever a new QueryTermService is selected by the user.
     */
    set activeTermService(queryTermService: QueryTermService | undefined) {
        this._activeTermService = queryTermService
        this.updateTable()
    }
    
    /**
     * Updates the table with neighbour terms data.
     * 
     * This function retrieves the table body element, clears existing rows, and then iterates over the neighbour terms of the active query term.
     * For each neighbour term, it creates a new row in the table and populates the cells with the term's value and hops.
     * 
     * @remarks
     * This function assumes that the table body element is already present in the HTML structure.
     */
    public updateTable(): void {
        // Get the table body element
        const tbody = this._dynamicTable.getElementsByTagName('tbody')[0]

        // Clear existing rows in the table
        tbody.innerHTML = '' 

        // Check if the activeTermService is defined
        if (this._activeTermService === undefined) return

        // Iterate over the neighbour terms of the active query term
        for(const neighbourTerm of this._activeTermService.visibleQueryTerm.neighbourTerms) {
            // Create a new row in the table
            const row = tbody.insertRow()

            // Create cells for the row
            const cell1 = row.insertCell(0)
            const cell2 = row.insertCell(1)

            // Set the text content of the cells
            cell1.innerHTML = neighbourTerm.value
            cell2.innerHTML = neighbourTerm.criteria;
        }
    }
}


class ResultsList {
    private _activeTermService: QueryTermService | undefined
    private readonly _dynamicList: HTMLElement

    /**
     * Constructor for the NeighbourTermsTable class.
     * Initializes the dynamicList property with the HTML element with the id 'resultsList'.
     */
    constructor() {
        this._dynamicList = document.getElementById('resultsList') as HTMLElement
    }

    /**
     * Sets the active QueryTermService and updates the list.
     *
     * @param queryTermService - The QueryTermService to be set as the active service.
     * This service will be used to retrieve and display data in the list.
     *
     * @returns {void} - This function does not return any value.
     *
     * @remarks
     * This function is responsible for setting the active QueryTermService and updating the table.
     * It is called whenever a new QueryTermService is selected by the user.
     */
    set activeTermService(queryTermService: QueryTermService | undefined) {
        this._activeTermService = queryTermService
        this.updateList()
    }

    /**
     * Updates the list of query results with new documents.
     * Clears existing list items in the results list before adding new ones.
     * 
     * @remarks
     * This function is responsible for populating the results list with the documents retrieved from the active query term.
     * It iterates over the documents in the ranking and creates list items for each one, including title and abstract elements.
     * Click event listeners and mouse event listeners are added to the title elements to handle user interactions.
     */
    public updateList(): void {
        // Clear existing list items
        this._dynamicList.innerHTML = '';

        // Check if the activeTermService is defined
        if (this._activeTermService === undefined) return

        // Get the ranking of the active query term
        const notExcludedDocuments = this._activeTermService.ranking.getNotExcludedDocuments()
    
        for (let i = 0; i < notExcludedDocuments.length; i++) {
            // Create a new list item, title and abstract elements
            const listItem = document.createElement('li');
            const titleElement = this._createTitleElement(i, notExcludedDocuments[i])
            const abstractElement = this._createAbstractElement(notExcludedDocuments[i])
    
            // Add a click event listener and mouse event listeners to the title element
            this._addEventListenersToTitleElement(titleElement, abstractElement)
    
            // Append the title and abstract to the list item
            listItem.appendChild(titleElement);
            listItem.appendChild(abstractElement);
    
            // Append the list item to the dynamic list container
            this._dynamicList.appendChild(listItem);
        }
    }

    /**
     * Creates a title element for a document.
     * 
     * @param index - The index of the document in the list.
     * @param doc - The document for which to create the title element.
     * 
     * @returns A new HTMLSpanElement representing the title of the document.
     */
    private _createTitleElement(index: number, doc: Document): HTMLSpanElement {
        const titleElement = document.createElement('span');
        titleElement.className = 'title';
        // Check if the sentences array is not empty
        const sentences = doc.sentences;
        const titleSentenceObject = sentences.length > 0 ? [sentences[0]] : [];
        // Highlight the title element with green color for the query terms and purple color for the neighbour terms
        titleElement.appendChild(document.createTextNode((index + 1) + '. '));
        const highlightedSpanContainer = this._applyHighlightingToSentences(titleSentenceObject);
        titleElement.appendChild(highlightedSpanContainer);
        return titleElement;
    }

    /**
     * Creates an abstract element for a document.
     * 
     * @param doc - The document for which to create the abstract element.
     * 
     * @returns A new HTMLParagraphElement representing the abstract of the document.
     */
    private _createAbstractElement(doc: Document): HTMLParagraphElement {
        const abstractElement = document.createElement('p');
        abstractElement.className = 'abstract';
        // Check if sentences exist and slice from them
        const sentences = doc.sentences;
        const abstractSentenceObjects = sentences.length > 0 ? sentences.slice(1) : [];
        // Highlight the abstract element with green color for the query terms and purple color for the neighbour terms
        const highlightedSpanContainer = this._applyHighlightingToSentences(abstractSentenceObjects);
        abstractElement.appendChild(highlightedSpanContainer);
        abstractElement.style.display = "none";
        return abstractElement;
    }

    /**
     * This function generates highlighted text for a given list of sentences.
     * It uses the query terms and neighbour terms to apply different colors to the words.
     *
     * @param sentenceObjects - An array of Sentence objects to generate highlighted text for.
     * @returns A span element containing the highlighted text for the given sentences.
     *
     * @remarks
     * The function retrieves the query terms and neighbour terms from the activeTermService,
     * and then applies highlighting to the words in the sentences based on these terms.
     * The highlighting is applied using different colors for the query terms and neighbour terms.
     */
    private _applyHighlightingToSentences(sentenceObjects: Sentence[]): HTMLSpanElement {
        const visibleQueryTerm = this._activeTermService?.visibleQueryTerm;
        if (visibleQueryTerm === undefined) return document.createElement('span');
        const queryTermsList = visibleQueryTerm.individualQueryTermsList;
        const userProximityTermsList = visibleQueryTerm.getNeighbourProximityTermsValues()
        const userFrequencyTermsList = visibleQueryTerm.getNeighbourFrequencyTermsValues()
        return this._getHighlightedText(sentenceObjects, queryTermsList, userProximityTermsList, userFrequencyTermsList);

    }

    /**
     * Applies highlighting to words in sentences based on query terms and neighbour terms.
     *
     * @param sentenceObjects - An array of Sentence objects to apply highlighting to.
     * @param queryTermsList - An array of strings representing the query terms.
     * @param userProximityTermsList - An array of strings representing the user proximity terms.
     * @param userFrequencyTermsList - An array of strings representing the user frequency terms.
     *
     * @returns A span element containing the highlighted sentences.
     *
     * @remarks
     * This function iterates over each sentence object, splits the text into words,
     * and applies highlighting based on the presence of query terms and neighbour terms.
     * The function uses regular expressions to match the query terms and neighbour terms.
     * The highlighted words are wrapped in HTML span tags with a specific background color.
     * The function then joins the highlighted words back into sentences and returns the result.
     */
    private _getHighlightedText(sentenceObjects: Sentence[], queryTermsList: string[], userProximityTermsList: string[], 
                                userFrequencyTermsList: string[]): HTMLSpanElement {
        if (sentenceObjects.length == 0) return document.createElement('span');
        let highlightedSentencesTextObject: [string, Sentence | undefined][] = []

        for (let sentenceObject of sentenceObjects) {
            const sentenceText = sentenceObject.rawText;
            if (sentenceObject.queryTerm.neighbourTerms.length == 0 || (userProximityTermsList.length == 0 && userFrequencyTermsList.length == 0)) {
                // If there are no user neighbour terms in the sentence, or no proximity/frequency terms in the user graph, then return the original sentence
                highlightedSentencesTextObject.push([sentenceText, undefined]);
            } else {
                // Else, split text by spaces and replace matching words
                const rawToProcessedMap = sentenceObject.rawToProcessedMap;
                const sentenceProximityTermsList = sentenceObject.queryTerm.getNeighbourProximityTermsValues()
                const highlightedSentenceText = this._getHighlightedSentence(sentenceText, rawToProcessedMap, queryTermsList, 
                    userProximityTermsList, userFrequencyTermsList, sentenceProximityTermsList);
                highlightedSentencesTextObject.push([highlightedSentenceText, sentenceObject]);
            }
        }

        return this._applyEventListenersToSentences(highlightedSentencesTextObject);
    }

    /**
     * Applies highlighting to words in a sentence based on query terms and neighbour terms.
     *
     * @param sentenceText - The raw text of the sentence.
     * @param rawToProcessedMap - A map that maps raw words to processed words in the sentence.
     * @param queryTermsList - An array of strings representing the query terms.
     * @param userProxTermsList - An array of strings representing the user proximity terms.
     * @param userFreqTermsList - An array of strings representing the user frequency terms.
     * @param sentenceProxTermsList - An array of strings representing the sentence proximity terms.
     *
     * @returns A string representing the highlighted sentence. The highlighted words are wrapped in HTML span tags with specific background colors.
     *
     * @remarks
     * The function iterates over each tuple in the rawToProcessedMap, determines the color based on the conditions, and highlights the word if a color is determined.
     * The highlighted words are then sorted based on their original indices, and the highlighted sentence is built by combining the highlighted parts.
     */
    private _getHighlightedSentence(sentenceText: string, rawToProcessedMap: RawToProcessedMap, queryTermsList: string[], 
                userProxTermsList: string[], userFreqTermsList: string[], sentenceProxTermsList: string[]): string {
        // Create an array to hold the final parts of the sentence
        let highlightedParts = this._buildHighlightedParts(rawToProcessedMap, queryTermsList, 
            userProxTermsList, userFreqTermsList, sentenceProxTermsList);

        // Sort the highlighted parts based on their original indices
        highlightedParts.sort((a, b) => a.firstIndex - b.firstIndex);

        // Build the highlighted sentence
        const resultSentence = this._buildHighlightedSentence(highlightedParts, sentenceText)

        return resultSentence;
    }

    /**
     * Builds an array of highlighted parts from a raw to processed map, based on query terms and user proximity/frequency terms.
     *
     * @param rawToProcessedMap - A map that maps raw words to processed words in a sentence.
     * @param queryTermsList - An array of strings representing the query terms.
     * @param userProxTermsList - An array of strings representing the user proximity terms.
     * @param userFreqTermsList - An array of strings representing the user frequency terms.
     * @param sentenceProxTermsList - An array of strings representing the sentence proximity terms.
     *
     * @returns An array of objects, where each object represents a highlighted part of the sentence.
     * Each object contains the first and last indices of the highlighted word, and the text of the highlighted word.
     * If a word is not highlighted, it is represented as a plain string.
     */
    private _buildHighlightedParts(rawToProcessedMap: RawToProcessedMap, queryTermsList: string[], 
        userProxTermsList: string[], userFreqTermsList: string[], sentenceProxTermsList: string[]
        ): { firstIndex: number; lastIndex: number; text: string }[] {
        // Create an array to hold the final parts of the sentence
        let highlightedParts: { firstIndex: number; lastIndex: number; text: string }[] = [];
        const processedWordsList = rawToProcessedMap.map(rawToProcessedTuple => rawToProcessedTuple[3]);
        const limitDistance = this._activeTermService?.visibleQueryTerm.hopLimit ?? 0;

        // Iterate over each tuple in rawToProcessedMap
        rawToProcessedMap.forEach(([firstIdx, lastIdx, rawWord, processedWord], wordIndex) => {
            let color = '';

            // Determine the color based on the conditions
            if (userProxTermsList.includes(processedWord) && sentenceProxTermsList.includes(processedWord)
                && this._hasNeighborMatchingQueryTerm(processedWordsList, queryTermsList, processedWord, wordIndex, limitDistance)) {
                color = '#98EE98'; // Green
            } else if (userFreqTermsList.includes(processedWord)) {
                color = '#B5BEF1'; // Blue
            } else if (queryTermsList.includes(processedWord)) {
                color = '#EEF373'; // Yellow
            }

            // Highlight the word if a color is determined
            if (color) {
                const highlightedWord = `<span style="background-color: ${color};">${rawWord}</span>`;
                highlightedParts.push({ firstIndex: firstIdx, lastIndex: lastIdx, text: highlightedWord });
            } else {
                highlightedParts.push({ firstIndex: firstIdx, lastIndex: lastIdx, text: rawWord });
            }
        });

        return highlightedParts;
    }

    /**
     * Checks if any neighbor of the processed word in a given list matches a query term.
     * 
     * @param processedWordsList - The list of processed terms to search in.
     * @param queryTermList - The list of query terms to match against.
     * @param processedWord - The word whose neighbors need to be checked.
     * @param wordIndex - The exact index of the processed word in rawToProcessedList.
     * @param distance - The maximum number of neighbors to check on each side (default is 4).
     * @returns `true` if any neighbor matches a query term, otherwise `false`.
     */
    private _hasNeighborMatchingQueryTerm(processedWordsList: string[], queryTermList: string[],
        processedWord: string, wordIndex: number, limitDistance: number): boolean {
        // Ensure the processedWord at the given wordIndex matches the list
        if (processedWordsList[wordIndex] !== processedWord) {
            console.log("The processedWord does not match the word at the given index.");
            return false;
        }

        // Calculate the range of indices to check
        const startIndex = Math.max(0, wordIndex - limitDistance);
        const endIndex = Math.min(processedWordsList.length - 1, wordIndex + limitDistance);

        // Iterate through the neighbors within the range
        for (let i = startIndex; i <= endIndex; i++) {
            if (i !== wordIndex && queryTermList.includes(processedWordsList[i])) {
                return true; // Found a match in the neighbors
            }
        }

        return false; // No match found
    }
    
    /**
     * Builds a highlighted sentence by combining the original sentence text with highlighted parts.
     *
     * @param highlightedParts - An array of tuples, where each tuple contains the first and last indices of a highlighted word,
     * and the text of the highlighted word.
     * @param sentenceText - The raw text of the sentence.
     *
     * @returns A string representing the highlighted sentence. The highlighted words are inserted into the sentence text
     * at their respective indices.
     */
    private _buildHighlightedSentence(highlightedParts: { firstIndex: number; lastIndex: number; text: string }[], 
        sentenceText: string): string {
        // Combine the sentence text with the highlighted words
        let resultSentence = '';
        let currentIndex = 0;

        highlightedParts.forEach(part => {
            // Append the text from the current index to the part's index
            resultSentence += sentenceText.substring(currentIndex, part.firstIndex);
            resultSentence += part.text;
            currentIndex = part.lastIndex + 1; // Move current index forward
        });

        // Append any remaining text
        resultSentence += sentenceText.substring(currentIndex);

        return resultSentence;
    }

    /**
     * Applies event listeners to sentences and returns a span element containing the highlighted sentences.
     *
     * @param highlightedSentences - An array of tuples, where each tuple contains a string representing a highlighted sentence
     * and a Sentence object representing the original sentence.
     *
     * @returns A span element containing the highlighted sentences with event listeners applied to the sentences.
     *
     * @remarks
     * The function creates a span element and adds spans to it, representing each highlighted sentence.
     * If the sentence contains query terms or neighbour terms, the function adds mouseenter and mouseleave event listeners
     * to the span element, changing the background color of the span and setting the visible sentence in the active term service.
     * The function also appends a separator (a dot followed by a space) between each sentence, except for the last sentence.
     */
    private _applyEventListenersToSentences(highlightedSentences: [string, Sentence | undefined][]): HTMLSpanElement {
        // Create the main container of type span
        const mainSpanContainer = document.createElement('span');

        // Add spans to the main container, and a separator in case it is not the last span
        highlightedSentences.forEach((sentence, index) => {
            const sentenceText = sentence[0];
            const sentenceObject = sentence[1];
            const spanElement = document.createElement('span');
            spanElement.innerHTML = sentenceText;

            // Add event listeners for mouseenter and mouseleave events if the sentence contains query terms or neighbour terms
            if (sentenceObject !== undefined) {
                spanElement.addEventListener("mouseenter", () => {
                    spanElement.style.backgroundColor = "#E4E4E4";
                    if (this._activeTermService !== undefined) this._activeTermService.visibleSentence = sentenceObject;
                });

                spanElement.addEventListener("mouseleave", () => {
                    spanElement.style.backgroundColor = "inherit";
                });
            }

            // Append the span element to the main container, with a separator if it is not the last span
            mainSpanContainer.appendChild(spanElement);
            if (index < (highlightedSentences.length - 1)) {
                mainSpanContainer.appendChild(document.createTextNode('. '));
            }
        });

        return mainSpanContainer;
    }

    /**
     * Adds event listeners to the title element and the abstract element.
     * 
     * @param titleElement - The HTMLSpanElement representing the title of a list item.
     * @param abstractElement - The HTMLParagraphElement representing the abstract of a list item.
     * 
     * @remarks
     * When the title element is clicked, the abstract element is displayed or hidden.
     * When the mouse hovers over the title element, the color changes to dark blue and the cursor becomes a pointer.
     * When the mouse leaves the title element, the color changes back to black.
     */
    private _addEventListenersToTitleElement(titleElement: HTMLSpanElement, abstractElement: HTMLParagraphElement): void {
        titleElement.addEventListener('click', () => {
            // When the list item is clicked, opens the original URL document webpage in a new tab
            //window.open('https://ieeexplore.ieee.org/document/' + documents[i].id, '_blank');
            if (abstractElement.style.display === "none") {
                abstractElement.style.display = "";
            } else {
                abstractElement.style.display = "none";
            }
        });

        titleElement.addEventListener("mouseenter", () => {
            titleElement.style.color = "darkblue";
            titleElement.style.cursor = "pointer";
        });

        titleElement.addEventListener("mouseleave", () => {
            titleElement.style.color = "black";
        });
    }
}


/**
 * Represents a component responsible for handling query input interactions.
 * 
 * This class is responsible for capturing user input from an HTML input element,
 * and sending the query to a query service when the Enter key is pressed.
 */
class QueryComponent {
    private readonly _queryService: QueryService
    private readonly _input: HTMLInputElement
    private readonly _searchIcon: HTMLElement
    private readonly _searchResultsInput: HTMLInputElement
    private readonly _limitDistanceInput: HTMLInputElement
    private readonly _graphTermsInput: HTMLInputElement
    private readonly _loadingBar: LoadingBar;

    /**
     * Constructs a new instance of QueryComponent.
     * This class is responsible for handling query input interactions.
     *
     * @param queryService - The QueryService instance to be used for sending queries.
     * @param loadingBar - The LoadingBar instance to manage the loading bar display.
     *
     * @remarks
     * The QueryComponent captures user input from an HTML input element,
     * and sends the query to the QueryService when the Enter key is pressed or the search icon is clicked.
     */
    constructor(queryService: QueryService, loadingBar: LoadingBar) {
        this._queryService = queryService
        this._input = document.getElementById('queryInput') as HTMLInputElement
        this._searchIcon = document.getElementById('searchIcon') as HTMLElement
        this._searchResultsInput = document.getElementById('searchResults') as HTMLInputElement;
        this._limitDistanceInput = document.getElementById('limitDistance') as HTMLInputElement;
        this._graphTermsInput = document.getElementById('graphTerms') as HTMLInputElement;
        this._loadingBar = loadingBar;

        // Set default values for the inputs
        this._searchResultsInput.value = "10";
        this._limitDistanceInput.value = "4";
        this._graphTermsInput.value = "5";

        this._addEventListeners()
    }

    /**
     * Handles the query input and adds event listeners to various elements.
     * 
     * This function adds event listeners for:
     * - "Enter" key presses in the query input field.
     * - Clicking the search icon.
     * - Toggling the visibility of search parameters.
     * - Validating and ensuring inputs stay within defined ranges.
     */
    private _addEventListeners(): void {
        // Event listener for "Enter" key presses
        this._input.addEventListener("keyup", event => {
            if(event.key === "Enter") {
                this._processQuery()
            }
        })

        // Event listener for clicking the search icon
        this._searchIcon.addEventListener("click", () => {
            this._processQuery()
        })

        // Add validation to ensure inputs stay within defined ranges
        this._addValidationListeners();
    }

    
    /**
     * Handles the validation of user input fields for search results, limit distance, and number of graph terms.
     * This function adds event listeners to the input fields to validate the input values and ensure they stay within defined ranges.
     *
     * @remarks
     * The function adds event listeners for the following input fields:
     * - `searchResultsInput`: Validates the number of search results. It should be at least 5.
     * - `limitDistanceInput`: Validates the limit distance for proximity-based queries. It should be between 2 and 10.
     * - `graphTermsInput`: Validates the number of graph terms to be considered. It should be between 1 and 20.
     */
    private _addValidationListeners(): void {
        // Add validation to ensure inputs stay within defined ranges
        // Validate search results input
        this._searchResultsInput.addEventListener("change", () => {
            let value = parseInt(this._searchResultsInput.value, 10);
            if (isNaN(value) || value < 5) {
                this._searchResultsInput.value = "5";
            }
        });

        // Validate limit distance input
        this._limitDistanceInput.addEventListener("change", () => {
            let value = parseInt(this._limitDistanceInput.value, 10);
            if (isNaN(value) || value < 2) {
                this._limitDistanceInput.value = "2";
            } else if (value > 10) {
                this._limitDistanceInput.value = "10";
            }
        });

        // Validate number of graph terms input
        this._graphTermsInput.addEventListener("change", () => {
            let value = parseInt(this._graphTermsInput.value, 10);
            if (isNaN(value) || value < 1) {
                this._graphTermsInput.value = "1";
            } else if (value > 20) {
                this._graphTermsInput.value = "20";
            }
        });
    }


    /**
     * Handles the query input and sends the query to the query service.
     * 
     * This function retrieves the trimmed input value from the query input field, clears the input field,
     * and checks if the value contains at least one alphanumeric character. If it does, it sends the query to the query service.
     * If the value is empty or does not contain any alphanumeric characters, it alerts the user to enter a valid query.
     */
    private async _processQuery(): Promise<void> {
        // Disable the input field to prevent further user interaction
        this._input.disabled = true;
        this._loadingBar.show();

        const queryValue = this._input.value.trim() // Get the trimmed input value
        this._input.value = '' // Clear the input field
        const alphanumericRegex = /[a-zA-Z0-9]/

        if (alphanumericRegex.test(queryValue)) {   // Check if the value contains at least one alphanumeric character
            const searchResults = parseInt(this._searchResultsInput.value, 10);
            const limitDistance = parseInt(this._limitDistanceInput.value, 10);
            const graphTerms = parseInt(this._graphTermsInput.value, 10);

            try {
                //Send the query to the query service
                await this._queryService.setQuery(queryValue, searchResults, limitDistance, graphTerms) 
            } catch (error) {
                console.error("Error processing query:", error);
                alert("Failed to process the query.");
            }
            
        } else if (queryValue !== '') {
            alert("Please enter a valid query.")    // Alert the user if the query is invalid
        }

        // Re-enable the input field after the process is complete, and end the loading bar
        this._input.disabled = false;
        this._loadingBar.hide();
    }
}


class RerankComponent {
    private readonly _queryService: QueryService
    private readonly _button: HTMLButtonElement
    private readonly _loadingBar: LoadingBar

    /**
     * Constructs a new instance of RerankComponent.
     * 
     * @param queryService - The QueryService instance to be used for reranking operations.
     * @param loadingBar - The LoadingBar instance to manage the loading bar display.
     */
    constructor(queryService: QueryService, loadingBar: LoadingBar) {
        this._queryService = queryService
        this._button = document.getElementById('rerankButton') as HTMLButtonElement
        this._loadingBar = loadingBar;

        // Add event listener to the button element
        this._button.addEventListener('click', this._handleRerankClick.bind(this))
    }

    /**
     * Handles the reranking button click event.
     * 
     * If the active QueryTermService exists, it creates a ranking object,
     * sends a POST request to the reranking endpoint, and handles the response.
     * 
     * @remarks
     * This method is asynchronous and uses the await keyword to handle the POST request.
     */
    private async _handleRerankClick(): Promise<void> {
        const activeTermService = this._queryService.activeQueryTermService;
        if (activeTermService === undefined) return;

        // Disable the button to prevent multiple clicks
        this._button.disabled = true;
        this._loadingBar.show();
        
        // Create the data to be sent in the POST request
        const ranking = activeTermService.ranking.toObject();
        console.log(ranking);

        try {
            // Send the POST request
            await this._queryService.setRerank(ranking);
        } catch (error) {
            console.error("An error occurred during reranking:", error);
            alert("Failed to process the rerank.");
        } finally {
            // Re-enable the button after the process is complete, and end the loading bar
            this._button.disabled = false;
            this._loadingBar.hide();
        }
    }
}


class LoadingBar {
    private readonly _element: HTMLElement;

    /**
     * Constructs a new instance of LoadingBar.
     * 
     * @param elementId - The ID of the HTML element representing the loading bar.
     */
    constructor() {
        const element = document.getElementById('loadingBar') as HTMLElement;
        this._element = element;
    }

    /**
     * Shows the loading bar by setting its display to "block".
     */
    public show(): void {
        this._element.style.display = "block";
    }

    /**
     * Hides the loading bar by setting its display to "none".
     */
    public hide(): void {
        this._element.style.display = "none";
    }
}


class CytoscapeManager {
    private static readonly _cyUser = cytoscape({
        container: document.getElementById("cy") as HTMLElement,
        layout: {
            name: "preset",
        },
        style: [
            {
                selector: '.' + NodeType.central_node,
                style: {
                "background-color": '#f0d200',
                'width': '20px',
                'height': '20px',
                'label': "data(id)",
                'font-size': '13px',
                'color': '#5d5d5d'
                },
            },
            {
                selector: "edge",
                style: {
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "line-color": "#ccc",
                "width": "2px", // set the width of the edge
                "font-size": "12px" // set the font size of the label            
                },
            },
            {
                selector: '.' + NodeType.outer_node,
                style: {
                  'background-color': '#73b201',
                  'width': '15px',
                  'height': '15px',
                  'label': 'data(label)',
                  'font-size': '12px',
                  'color': '#4b4b4b'
                }
            }
        ],
        userZoomingEnabled: false,
        userPanningEnabled: false
    });

    private static readonly _cySentence = cytoscape({
        container: document.getElementById("cySentence") as HTMLElement,
        layout: {
            name: "preset",
        },
        style: [
            {
                selector: '.' + NodeType.central_node,
                style: {
                "background-color": '#f0d200',
                'width': '16px',
                'height': '16px',
                'label': "data(id)",
                'font-size': '10px',
                'color': '#5d5d5d'
                },
            },
            {
                selector: "edge",
                style: {
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "line-color": "#ccc",
                "width": "2px", // set the width of the edge
                "font-size": "11px" // set the font size of the label            
                },
            },
            {
                selector: '.' + NodeType.outer_node,
                style: {
                    'background-color': '#73b201',
                    'width': '12px',
                    'height': '12px',
                    'label': 'data(label)',
                    'font-size': '9px',
                    'color': '#4b4b4b'
                }
            }
        ],
        userZoomingEnabled: true,
        userPanningEnabled: true
    });

    public static getCyUserInstance(): cytoscape.Core {
        return this._cyUser;
    }

    public static getCySentenceInstance(): cytoscape.Core {
        return this._cySentence;
    }
}



//// CYTOSCAPE CONFIGURATION

// When the user drags a node
CytoscapeManager.getCyUserInstance().on('drag', 'node', evt => {
    queryService.activeQueryTermService?.nodeDragged(evt.target.id(), evt.target.position())
})

// When the user right-clicks a node
CytoscapeManager.getCyUserInstance().on('cxttap', "node", evt => {
    queryService.activeQueryTermService?.removeVisibleNeighbourTerm(evt.target.id())
});

// When the user right-clicks a edge
CytoscapeManager.getCyUserInstance().on('cxttap', "edge", evt => {
    queryService.activeQueryTermService?.removeVisibleNeighbourTerm(evt.target.id().substring(2))
});

// When the user hovers it's mouse over a node
CytoscapeManager.getCyUserInstance().on('mouseover', 'node', (evt: cytoscape.EventObject) => {
    queryService.activeQueryTermService?.changeCursorType(evt.target.id(), 'pointer');
});

// When the user moves it's mouse away from a node
CytoscapeManager.getCyUserInstance().on('mouseout', 'node', (evt: cytoscape.EventObject) => {
    queryService.activeQueryTermService?.changeCursorType(evt.target.id(), 'default');
});


const loadingBar = new LoadingBar();
const queryService = new QueryService();
const queryComponent = new QueryComponent(queryService, loadingBar);
const rerankComponent = new RerankComponent(queryService, loadingBar);


CytoscapeManager.getCyUserInstance().ready(() => {})


// quick way to get instances in console
;(window as any).cy = CytoscapeManager.getCyUserInstance()
;(window as any).queryService = queryService
