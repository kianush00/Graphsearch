import cytoscape from "cytoscape";

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
        const pos1 = node1.getPosition()
        const pos2 = node2.getPosition()

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

    /**
     * Separates a boolean query into individual terms.
     *
     * @param query - The boolean query to be separated.
     * @returns An array of strings representing the separated terms.
     *
     * @remarks
     * This function uses a regular expression to match boolean operators, parentheses, colons, and terms.
     * It then filters the matches, removing specified elements such as parentheses and boolean operators.
     * If no matches are found, an empty array is returned.
     */
    public static separateBooleanQuery(query: string): string[] {
        // Defines a regular expression to match boolean operators, parentheses, colons, and terms
        const pattern = /\bAND\b|\bOR\b|\bNOT\b|\(|\)|\w+|:/g;
        
        // Finds all matches using the regular expression
        let tokens = query.match(pattern);

        // Defines a set of elements to remove
        const elementsToRemove = new Set(['(', ')', 'AND', 'OR', 'NOT']);

        // Filters the array, removing the specified elements
        if (tokens !== null) {
            return tokens.filter(item => !elementsToRemove.has(item));
        }

        return []
    }
}


class ConversionUtils {
    private static minDistance: number = 50.0
    private static maxDistance: number = 140.0
    private static hopMinValue: number = 1.0

    /**
     * Converts the number of hops to the corresponding distance in the graph.
     *
     * @param hops - The number of hops from the central node to the neighbour term.
     * @param hopMaxValue - The maximum number of hops allowed in the graph.
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
    public static convertHopsToDistance(hops: number, hopMaxValue: number): number {
        if (hopMaxValue < 2) return 1.0
        const normalizedValue = this.normalize(hops, this.hopMinValue, hopMaxValue, this.minDistance, this.maxDistance)
        return normalizedValue
    }

    /**
     * Converts a given distance to hops, based on a normalized value within a specified range.
     *
     * @param distance - The distance to be converted to hops.
     * @param hopMaxValue - The maximum number of hops that can be achieved.
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
        const normalizedValue = this.normalize(distance, this.minDistance, this.maxDistance, this.hopMinValue, hopMaxValue)
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
        return distance < this.minDistance || distance > this.maxDistance
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
                mode: 'cors',
                method: 'POST',
                headers: {
                'Content-Type': 'application/json',
                'Origin': 'https://localhost:3000',
                'Access-Control-Allow-Origin': 'http://localhost:8080'
                },
                body: JSON.stringify(data),
            })
        
            // Handle the response
            const result = await response.json()
            console.log('Success:', result)
            const sizeKb = new TextEncoder().encode(JSON.stringify(result)).length / 1024
            console.log('Size of response in KB:', sizeKb)
            return result
        } catch (error) {
            console.error('Error:', error)
        }
    }

}



interface EdgeData {
    id: string
    source: string
    target: string
    distance: number
}


class Edge {
    private id: string
    private sourceNode: GraphNode
    private targetNode: GraphNode
    private distance: number
    private hopLimit: number

    /**
     * Represents an edge in the graph, connecting two nodes.
     * 
     * @param sourceNode - The source node of the edge.
     * @param targetNode - The target node of the edge.
     * @param hopLimit - The maximum number of hops allowed for the edge.
     */
    constructor(sourceNode: GraphNode, targetNode: GraphNode, hopLimit: number) {
        this.id = "e_" + targetNode.getId()
        this.sourceNode = sourceNode
        this.targetNode = targetNode
        this.distance = MathUtils.getDistanceBetweenNodes(sourceNode, targetNode)
        this.hopLimit = hopLimit
        cy.add(this.toObject())
    }

    /**
     * Sets the distance for the edge between the source and target nodes.
     * This method updates the distance value and the corresponding edge data in the graph.
     *
     * @param distance - The new distance for the edge.
     *
     * @remarks
     * This method assumes that the source and target nodes are already associated with the edge.
     * It also assumes that the graph is represented using the Cytoscape.js library.
     *
     * @returns {void} - This function does not return any value.
     */
    public setDistance(distance: number): void {
        this.distance = distance
        let cyEdge = cy.edges(`[source = "${this.sourceNode.getId()}"][target = "${this.targetNode.getId()}"]`)
        const hops = ConversionUtils.convertDistanceToHops(this.distance, this.hopLimit)
        cyEdge.data('distance', hops)
    }

    /**
     * Updates the distance between the source and target nodes.
     *
     * This function calculates the distance between the source and target nodes using the `MathUtils.getDistanceBetweenNodes` method.
     * It then sets the calculated distance as the distance between the nodes using the `this.setDistance` method.
     *
     * @remarks
     * This function assumes that the source and target nodes are already defined and valid.
     */
    public updateDistance(): void {
        this.setDistance(MathUtils.getDistanceBetweenNodes(this.sourceNode, this.targetNode))
    }

    public getDistance(): number {
        return this.distance
    }

    /**
     * Removes the visual node from the graph interface.
     * 
     * This function removes the node with the given ID from the graph interface.
     * It uses the Cytoscape.js library to select the node by its ID and remove it from the graph.
     *
     * @remarks
     * This function should be called when the node is no longer needed in the graph interface.
     * It ensures that the node is removed from the visual representation of the graph.
     */
    public remove(): void {
        cy.remove(cy.getElementById(this.id))
    }

    /**
     * Converts the Edge instance into a serializable object.
     * 
     * @returns An object containing the data of the Edge.
     * The object has properties: id, source, target, and distance.
     * The distance is converted to hops using the ConversionUtils.
     */
    public toObject(): { data: EdgeData } {
        return {
            data: {
                id: this.id,
                source: this.sourceNode.getId(),
                target: this.targetNode.getId(),
                distance: ConversionUtils.convertDistanceToHops(this.distance, this.hopLimit)
            },
        }
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
    id: string
    label: string
}

class GraphNode {
    protected id: string
    protected label: string
    protected position: Position
    protected type: NodeType
    
    /**
     * Represents a node in the graph.
     * 
     * @param id - The unique identifier of the node.
     * @param label - The label of the node.
     * @param position - The position of the node in the graph.
     * @param type - The type of the node.
     */
    constructor(id: string, label: string, position: Position, type: NodeType) {
        this.id = id
        this.label = label
        this.position = position
        this.type = type
    }

    public getId(): string {
        return this.id
    }

    public getPosition(): Position { 
        return this.position
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
    public setLabel(label: string): void {
        this.label = label
        cy.getElementById(this.id).data('label', label)
    }

    /**
     * Removes the visual node from the graph interface.
     *
     * @remarks
     * This function removes the node with the given ID from the graph interface.
     * It uses the Cytoscape.js library to select the node by its ID and remove it from the graph.
     */
    public remove(): void {
        cy.remove(cy.getElementById(this.id))
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
    public toObject(): { data: NodeData; position: Position } {
        return {
            data: {
                id: this.getId(),
                label: this.label,
            },
            position: this.position,
        }
    }
}


class CentralNode extends GraphNode {
    /**
     * Represents a central node in the graph.
     * It manages the associated views, such as the visual node in the graph interface.
     * 
     * @param id - The unique identifier of the central node.
     * @param x - The x-coordinate of the central node's position in the graph.
     * @param y - The y-coordinate of the central node's position in the graph.
     */
    constructor(id: string, x: number, y: number) {
        let _id = id
        let _label = id
        let _position = { x, y }
        let _type = NodeType.central_node
        super(_id, _label, _position, _type)
        this.addVisualNodeToInterface()
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
    private addVisualNodeToInterface(): void {
        cy.add(this.toObject()).addClass(this.type.toString()).lock().ungrabify()
    }
}


class OuterNode extends GraphNode {
    /**
     * Represents an outer node in the graph.
     * It manages the associated views, such as the visual node in the graph interface.
     * 
     * @param id - The unique identifier of the outer node.
     * @param distance - The distance from the central node to the outer node.
     *                    Default value is 0, which means the outer node will be positioned randomly.
     */
    constructor(id: string, distance: number = 0) {
        let _id = id
        let _label = id
        // Generates a random angle from the provided distance, to calculate the new position
        let _position = MathUtils.getAngularPosition(MathUtils.getRandomAngle(), distance)
        let _type = NodeType.outer_node
        super(_id, _label, _position, _type)
        this.addVisualNodeToInterface()
    }

    /**
     * Sets the position of the OuterNode in the graph.
     *
     * @param position - The new position for the OuterNode.
     *
     * @remarks
     * This function updates the position of the OuterNode and also updates the position in the graph interface.
     */
    public setPosition(position: Position): void {
        this.position = position;
        this.updateVisualPosition();
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
        this.position = MathUtils.getAngularPosition(angle, this.getDistance());
        this.updateVisualPosition();
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
        this.position = MathUtils.getAngularPosition(MathUtils.getRandomAngle(), distance);
        this.updateVisualPosition();
    }

    /**
     * Calculates the distance from the central node to the OuterNode.
     *
     * @returns {number} - The distance from the central node to the OuterNode.
     *
     * @remarks
     * This function calculates the distance from the central node to the OuterNode using the Euclidean distance formula.
     */
    private getDistance(): number {
        return MathUtils.calculateEuclideanDistance(this.position.x, this.position.y);
    }

    /**
     * Updates the visual position of the OuterNode in the graph interface.
     *
     * @remarks
     * This function is responsible for updating the position of the OuterNode in the graph interface.
     * It sets the position of the OuterNode to the current position of the OuterNode instance.
     */
    private updateVisualPosition(): void {
        cy.getElementById(this.id).position(this.position)
    }

    /**
     * Adds a visual node to the graph interface.
     * 
     * This function creates a new node in the graph using the `toObject` method,
     * which returns an object containing the node's data and position.
     * The node is then added to the graph using the `cy.add` method.
     * The node's class is set to the string representation of the node's type.
     */
    private addVisualNodeToInterface(): void {
        cy.add(this.toObject()).addClass(this.type.toString())
    }
}


/**
 * Represents a term in a graph.
 * It is associated with a graph node.
 */
class Term {
    protected value: string
    protected node: GraphNode | undefined

    /**
     * Represents a term in a graph.
     * It is associated with a graph node.
     */
    constructor(value: string) {
        this.value = value
    }

    /**
     * Sets the label of the term and updates the associated graph node.
     * 
     * @param value - The new label for the term.
     */
    public setLabel(value: string): void {
        this.value = value
        this.node?.setLabel(value)
    }

    public getValue(): string {
        return this.value
    }

    public getNode(): GraphNode | undefined {
        return this.node
    }
}


interface ViewManager {
    displayViews(): void
    removeViews(): void
}

interface NTermObject {
    term: string;
    ponderation: number;
    distance: number;
}


class NeighbourTerm extends Term implements ViewManager {
    protected node: OuterNode | undefined
    private queryTerm: QueryTerm
    private hops: number
    private nodePosition: Position = { x: 0, y: 0 }
    private edge: Edge | undefined
    private ponderation: number
    private hopLimit: number

    /**
     * Represents a neighbour term in the graph.
     * It manages the associated views, such as the OuterNode and Edge.
     * 
     * @param queryTerm - The QueryTerm associated with this neighbour term.
     * @param value - The value of the neighbour term.
     * @param hops - The number of hops from the central node to this neighbour term.
     * @param ponderation - The ponderation of this neighbour term.
     * @param hopLimit - The maximum number of hops allowed for the neighbour term.
     */
    constructor(queryTerm: QueryTerm, value: string, hops: number, ponderation: number, hopLimit: number) {
        super(value)
        this.queryTerm = queryTerm
        this.ponderation = ponderation
        this.hops = hops
        this.hopLimit = hopLimit
        this.setLabel(value)
    }

    /**
     * Displays the views of the neighbour term in the graph.
     * This includes creating and positioning the OuterNode and Edge.
     */
    public displayViews(): void {
        this.node = new OuterNode(TextUtils.getRandomString(24))
        this.node.setPosition(this.nodePosition)
        this.node.setLabel(this.value)
        if (this.queryTerm.getNode() === undefined) return 
        this.edge = new Edge(this.queryTerm.getNode() as CentralNode, this.node, this.hopLimit)
    }

    /**
     * Removes the views of the neighbour terms and the central node.
     * 
     * This function is responsible for removing the visual nodes (OuterNodes and CentralNode)
     * and edges (connecting the CentralNode to the OuterNodes) from the graph interface.
     */
    public removeViews(): void {
        this.node?.remove()
        this.edge?.remove()
    }

    public getHops(): number {
        return this.hops
    }

    public getPonderation(): number {
        return this.ponderation
    }

    public getHopLimit(): number {
        return this.hopLimit
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
        return {
            term: this.value,
            ponderation: this.ponderation,
            distance: this.hops
        }
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
    public setPosition(position: Position): void {
        const nodeDistance = this.edge?.getDistance() ?? 0
        this.nodePosition = this.validatePositionWithinRange(position, nodeDistance)
        const distance = MathUtils.calculateEuclideanDistance(this.nodePosition.x, this.nodePosition.y)
        this.hops = ConversionUtils.convertDistanceToHops(distance, this.hopLimit)
        this.updateNodePosition(distance)
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
        const newAngle = (index / neighbourTermsLength) * Math.PI * 2
        const nodeDistance = ConversionUtils.convertHopsToDistance(this.hops, this.hopLimit)
        this.nodePosition = MathUtils.getAngularPosition(newAngle, nodeDistance)
        this.updateNodePosition(nodeDistance)
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
    private validatePositionWithinRange(position: Position, nodeDistance: number): Position {
        let positionDistance = MathUtils.calculateEuclideanDistance(position.x, position.y)

        if (this.edge !== undefined && this.node !== undefined ) {
            if (ConversionUtils.validateDistanceOutOfRange(positionDistance)) {
                let angle = Math.atan2(position.y, position.x)
                let adjustedX = Math.cos(angle) * nodeDistance
                let adjustedY = Math.sin(angle) * nodeDistance
                position.x = adjustedX
                position.y = adjustedY
            }
        }
        return position
    }

    /**
     * Updates the position of the neighbour term node and the neighbour term's hops.
     *
     * @param distance - The distance from the central node to the neighbour term node.
     * This distance is used to update the neighbour term's hops and the position of the neighbour term node.
     *
     * @returns {void} - This function does not return any value.
     */
    private updateNodePosition(distance: number): void {
        this.node?.setPosition(this.nodePosition)
        this.edge?.setDistance(distance)
    }
}


/**
 * Represents a query term that is associated with a central node in the graph.
 * It also manages neighbour terms related to the query term.
 */
class QueryTerm extends Term implements ViewManager {
    protected node: CentralNode | undefined
    private neighbourTerms: NeighbourTerm[] = []

    /**
     * Displays the views of the query term and its associated neighbour terms in the graph.
     * This includes creating and positioning the CentralNode and OuterNodes.
     */
    public displayViews(): void {
        this.node = new CentralNode(this.value, 0, 0)
        for (let neighbourTerm of this.neighbourTerms) {
            neighbourTerm.displayViews()
        }
    }

    /**
    * Removes the views of the neighbour terms and the central node.
    */
    public removeViews(): void {
        for (let neighbourTerm of this.neighbourTerms) {
            neighbourTerm.removeViews()
        }
        this.node?.remove()
    }

    public getNeighbourTerms(): NeighbourTerm[] {
        return this.neighbourTerms
    }

    public setNeighbourTerms(neighbourTerms: NeighbourTerm[]): void {
        this.neighbourTerms = neighbourTerms
    }

    public getNeighbourTermsValues(): string[] {
        return this.neighbourTerms.map(term => term.getValue())
    }

    public getNeighbourTermsAsObjects(): NTermObject[] {
        return this.neighbourTerms.map(term => term.toObject())
    }

    public getNeighbourTermByNodeId(id: string): NeighbourTerm | undefined {
        return this.neighbourTerms.find(nterm => nterm.getNode()?.getId() === id)
    }

    public getNeighbourTermByValue(value: string): NeighbourTerm | undefined {
        return this.neighbourTerms.find(nterm => nterm.getValue() === value)
    }

    public addNeighbourTerm(neighbourTerm: NeighbourTerm): void {
        this.neighbourTerms.push(neighbourTerm)
        this.updateOuterNodesAngles()
    }

    public removeNeighbourTerm(neighbourTerm: NeighbourTerm): void {
        this.neighbourTerms = this.neighbourTerms.filter(term => term !== neighbourTerm)
        neighbourTerm.removeViews()
        this.updateOuterNodesAngles()
    }

    private updateOuterNodesAngles(): void {
        for (let i = 0; i < this.neighbourTerms.length; i++) {
            this.neighbourTerms[i].updateSymmetricalAngularPosition(this.neighbourTerms.length, i)
        }
    }
}


class TextElement {
    protected queryTerm: QueryTerm

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
        this.queryTerm = new QueryTerm(queryTermValue)
        this.initializeNeighbourTermsFromResponse(responseNeighbourTerms, hopLimit)
    }

    public getQueryTerm(): QueryTerm {
        return this.queryTerm
    }

    /**
     * Initializes neighbour terms from the response data.
     * 
     * @param responseNeighbourTerms - An array of objects containing neighbour term data retrieved from the response.
     * Each object has properties: term, distance, and ponderation.
     * 
     * @param hopLimit - The maximum number of hops allowed for the neighbour terms in the document.
     * 
     * @returns {void} - This function does not return any value.
     */
    private initializeNeighbourTermsFromResponse(responseNeighbourTerms: any[], hopLimit: number): void {
        const neighbourTerms = []
        for (const termObject of responseNeighbourTerms) {
            let neighbourTerm = new NeighbourTerm(this.queryTerm, termObject.term, 
                termObject.distance, termObject.ponderation, hopLimit)
            neighbourTerms.push(neighbourTerm)
        }
        this.queryTerm.setNeighbourTerms(neighbourTerms)
    }
}



interface SentenceObject {
    position_in_doc: number;
    raw_text: string;
    neighbour_terms: NTermObject[];
}

class Sentence extends TextElement {
    private positionInDoc: number
    private rawText: string

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
    ​ */
    constructor(queryTermValue: string, responseNeighbourTerms: any[], hopLimit: number, positionInDoc: number, rawText: string){
        super(queryTermValue, responseNeighbourTerms, hopLimit)
        this.positionInDoc = positionInDoc
        this.rawText = rawText
    }

    public getPositionInDoc(): number {
        return this.positionInDoc
    }

    public getRawText(): string {
        return this.rawText
    }

    public toObject(): SentenceObject {
        return {
            position_in_doc: this.positionInDoc,
            raw_text: this.rawText,
            neighbour_terms: this.queryTerm.getNeighbourTermsAsObjects()
        }
    }
}


interface DocumentObject {
    doc_id: string;
    title: string;
    abstract: string;
    weight: number;
    neighbour_terms: NTermObject[];
    sentences: SentenceObject[];
}

class Document extends TextElement{
    private id: string
    private title: string
    private abstract: string
    private weight: number
    private sentences: Sentence[] = []

    /**
    ​ * Constructor for the Document class.
    ​ * Initializes a new Document instance with the provided query term value, neighbour terms, document details, and sentence data.
    ​ *
    ​ * @param queryTermValue - The value of the query term associated with the document.
    ​ * @param responseNeighbourTerms - An array of objects representing neighbour terms retrieved from the response.
    ​ * Each object has properties: term, distance, and ponderation.
    ​ * @param hopLimit - The maximum number of hops allowed for the neighbour terms in the document.
    ​ * @param idTitleAbstract - An array containing the document's id, title, and abstract.
    ​ * @param weight - The weight of the document.
    ​ * @param responseSentences - An array of objects representing sentences retrieved from the response.
    ​ * Each object has properties: position_in_doc, raw_text, and neighbour_terms.
    ​ */
    constructor(queryTermValue: string, responseNeighbourTerms: any[], hopLimit: number, idTitleAbstract: [string, string, string], 
        weight: number, responseSentences: any[]){
        super(queryTermValue, responseNeighbourTerms, hopLimit)
        this.id = idTitleAbstract[0]
        this.title = idTitleAbstract[1]
        this.abstract = idTitleAbstract[2]
        this.weight = weight
        this.sentences = this.initializeSentencesFromResponse(responseSentences, hopLimit)
    }

    public getId(): string {
        return this.id
    }

    public getTitle(): string {
        return this.title
    }

    public getAbstract(): string {
        return this.abstract
    }

    public getWeight(): number {
        return this.weight
    }

    public getSentences(): Sentence[] {
        return this.sentences
    }

    public toObject(): DocumentObject {
        return {
            doc_id: this.id,
            title: this.title,
            abstract: this.abstract,
            weight: this.weight,
            neighbour_terms: this.queryTerm.getNeighbourTermsAsObjects(),
            sentences: this.sentences.map(sentence => sentence.toObject())
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
    private initializeSentencesFromResponse(responseSentences: any[], hopLimit: number): Sentence[] {
        const sentences = []
        for (const sentenceObject of responseSentences) {
            let sentence = new Sentence(this.queryTerm.getValue(), sentenceObject.neighbour_terms, 
                    hopLimit, sentenceObject.position_in_doc, sentenceObject.raw_text)
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
    private visibleQueryTerm: QueryTerm
    private completeQueryTerm: QueryTerm
    private documents: Document[] = []

    constructor(queryTermValue: string) {
        this.visibleQueryTerm = new QueryTerm(queryTermValue)
        this.completeQueryTerm = new QueryTerm(queryTermValue)
    }

    public getVisibleQueryTerm(): QueryTerm {
        return this.visibleQueryTerm
    }

    public getCompleteQueryTerm(): QueryTerm {
        return this.completeQueryTerm
    }

    public getDocuments(): Document[] {
        return this.documents
    }

    public addDocument(document: Document): void {
        this.documents.push(document)
    }

    /**
     * Reorders the documents in the ranking based on the provided positions array.
     * If the lengths of the positions array and the documents array do not match,
     * logs an error message to the console and returns without modifying the documents.
     *
     * @param positions - An array of integers representing the new order of the documents.
     * Each integer corresponds to the index of a document in the documents array.
     *
     * @remarks
     * This function iterates over the positions array and creates a new array of documents
     * in the new order specified by the positions array.
     * It then assigns the reorderedDocuments array back to the documents property of the Ranking instance.
     */
    public reorderDocuments(positions: number[]): void {
        if (positions.length !== this.documents.length) {
            console.log('Positions array length must match documents array length.');
            return
        }
        const reorderedDocuments = new Array(this.documents.length);
        for (let i = 0; i < positions.length; i++) {
            reorderedDocuments[i] = this.documents[positions[i]];
        }
        this.documents = reorderedDocuments;
    }

    public toObject(): RankingObject {
        return {
            visible_neighbour_terms: this.visibleQueryTerm.getNeighbourTermsAsObjects(),
            documents: this.documents.map(document => document.toObject())
        }
    }
}


/**
 * A service class responsible for managing query terms and their associated data.
 */
class QueryTermService {
    private queryService: QueryService
    private ranking: Ranking
    private isVisible: boolean = false

    constructor(queryService: QueryService, queryTermValue: string, searchResults: number, limitDistance: number, graphTerms: number) {
        this.queryService = queryService
        this.ranking = new Ranking(queryTermValue)
        this.retrieveData(searchResults, limitDistance, graphTerms)
    }

    public getVisibleQueryTerm(): QueryTerm {
        return this.ranking.getVisibleQueryTerm()
    }

    public getCompleteQueryTerm(): QueryTerm {
        return this.ranking.getCompleteQueryTerm()
    }

    public getRanking(): Ranking {
        return this.ranking
    }

    /**
     * If the node is dragged, updates the position of the neighbour term node and 
     * updates the neighbour term's hops.
     * @param id - The id of the neighbour term node.
     * @param position - The new position of the neighbour term node.
     */
    public nodeDragged(id: string, position: Position): void {
        let neighbourTerm = this.getVisibleQueryTerm().getNeighbourTermByNodeId(id)
        if (neighbourTerm === undefined) return
        neighbourTerm.setPosition(position)

        // Update the neighbour terms table with the new hops values
        this.queryService.updateNeighbourTermsTable()
    }

    /**
     * Displays the QueryTerm and its associated views in the graph.
     * This includes creating and positioning the CentralNode and OuterNodes.
     */
    public display(): void {
        this.isVisible = true // Mark the QueryTerm as visible

        // Remove any existing views associated with the QueryTerm
        this.getVisibleQueryTerm().removeViews()

        // Display the views associated with the QueryTerm
        this.getVisibleQueryTerm().displayViews()

        // Center the graph on the CentralNode
        this.center()
    }

    /**
     * This method removes the visual nodes and edges from the graph interface.
     */
    public deactivate(): void {
        this.isVisible = false
        this.getVisibleQueryTerm().removeViews()
    }

    /**
     * Adds a neighbour term to the QueryTerm's neighbour terms list.
     * It also updates the neighbour terms table in the QueryService.
     * If the QueryTerm is currently visible, it displays the views of the neighbour term.
     *
     * @param neighbourTerm - The neighbour term to be added.
     */
    public addVisibleNeighbourTerm(neighbourTerm: NeighbourTerm): void {
        this.getVisibleQueryTerm().addNeighbourTerm(neighbourTerm)
        this.queryService.updateNeighbourTermsTable()
        this.queryService.updateAddTermsTable()
        if (this.isVisible) this.display()
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
        let neighbourTerm = this.getVisibleQueryTerm().getNeighbourTermByNodeId(id)
        if (neighbourTerm === undefined) return
        this.getVisibleQueryTerm().removeNeighbourTerm(neighbourTerm)
        this.queryService.updateNeighbourTermsTable()
        this.queryService.updateAddTermsTable()
        if (this.isVisible) this.display()
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
        this.getCompleteQueryTerm().addNeighbourTerm(neighbourTerm)
    }

    /**
     * Centers the graph on the CentralNode.
     * 
     * This function is responsible for zooming in the graph and centering it on the CentralNode.
     * It first zooms in the graph by a factor of 1.2, then checks if the visible query term has a node.
     * If the node exists and is a CentralNode, it centers the graph on the node.
     */
    private center(): void {
        cy.zoom(1.2)
        if (this.getVisibleQueryTerm().getNode() === undefined) return 
        cy.center(cy.getElementById((this.getVisibleQueryTerm().getNode() as CentralNode).getId()))
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
    private async retrieveData(searchResults: number, limitDistance: number, graphTerms: number): Promise<void> {
        // Define the endpoint for retrieving neighbour terms data
        const endpoint = 'get-ranking'

        // Send a POST request to the endpoint with the query term value
        let _query = this.getVisibleQueryTerm().getValue()
        const data = {query: _query, search_results: searchResults, limit_distance: limitDistance, graph_terms: graphTerms}
        const result = await HTTPRequestUtils.postData(endpoint, data)

        // Check if the result is not null
        if (result) {
            this.generateVisibleNeighbourTerms(result, limitDistance)
            this.generateCompleteNeighbourTerms(result, limitDistance)
            this.generateRankingDocuments(result, limitDistance)
        }
    }

    /**
     * Generates visible neighbour terms for the current query term.
     * 
     * @param result - The result object containing neighbour terms data.
     * The result object is expected to have a property 'visible_neighbour_terms',
     * which is an array of objects representing neighbour terms.
     * Each object should have properties 'term', 'distance', and 'ponderation'.
     * 
     * @param hopLimit - The maximum number of hops allowed for the neighbour terms.
     * 
     * @returns {void} - This function does not return any value.
     */
    private generateVisibleNeighbourTerms(result: any, hopLimit: number): void {
        // Iterate over the neighbour terms in the result
        for (let termObject of result['visible_neighbour_terms']) {
            // Create a new NeighbourTerm instance for each term object
            let neighbourTerm = this.initializeNewNeighbourTerm(termObject, hopLimit)

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
     * @param hopLimit - The maximum number of hops allowed for the neighbour terms.
     * 
     * @returns {void} - This function does not return any value.
     * It iterates over the neighbour terms in the result, creates a new NeighbourTerm instance for each term object,
     * and adds the neighbour term to the complete QueryTerm's neighbour terms list.
     */
    private generateCompleteNeighbourTerms(result: any, hopLimit: number): void {
        // Iterate over the neighbour terms in the result
        for (let termObject of result['complete_neighbour_terms']) {
            // Create a new NeighbourTerm instance for each term object
            let neighbourTerm = this.initializeNewNeighbourTerm(termObject, hopLimit)

            // Add the neighbour term to the complete QueryTerm's neighbour terms list
            this.addCompleteNeighbourTerm(neighbourTerm)
        }

        // Update the neighbour terms table in the QueryService
        this.queryService.updateAddTermsTable()
    }

    /**
     * Initializes a new NeighbourTerm instance based on the provided term object and hop limit.
     * 
     * @param termObject - An object containing properties 'term', 'distance', and 'ponderation' representing a neighbour term.
     * @param hopLimit - The maximum number of hops allowed for the neighbour terms.
     * 
     * @returns A new NeighbourTerm instance with the provided term value, distance, ponderation, and hop limit.
     */
    private initializeNewNeighbourTerm(termObject: any, hopLimit: number): NeighbourTerm {
        return new NeighbourTerm(this.getVisibleQueryTerm(), termObject.term, 
                    termObject.distance, termObject.ponderation, hopLimit)
    }

    /**
     * Generates ranking documents for the current query term, and updates the results list component.
     * 
     * @param result - The result object containing ranking documents data.
     * The result object is expected to have a property 'documents',
     * which is an array of objects representing documents.
     * Each object should have properties 'doc_id', 'title', 'abstract', and 'neighbour_terms'.
     * 
     * @param hopLimit - The maximum number of hops allowed for the neighbour terms in the documents.
     * 
     * @returns {void} - This function does not return any value.
     */
    private generateRankingDocuments(result: any, hopLimit: number): void {
        // Iterate over the documents in the result
        for (let documentObject of result['documents']) {
            const doc_id = documentObject['doc_id']
            const title = documentObject['title']
            const abstract = documentObject['abstract']
            const weight = documentObject['weight']
            const response_neighbour_terms = documentObject['neighbour_terms']
            const sentences = documentObject['sentences']
            let document = new Document(this.ranking.getVisibleQueryTerm().getValue(), response_neighbour_terms, hopLimit, 
                    [doc_id, title, abstract], weight, sentences)
            this.addDocument(document)
        }

        // Update the ranking's documents list
        this.queryService.updateResultsList()
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
    private addDocument(document: Document): void {
        this.getRanking().addDocument(document)
    }
}


class QueryService {
    private activeQueryTermService: QueryTermService | undefined
    private queryTermServices: QueryTermService[]
    private neighbourTermsTable: NeighbourTermsTable
    private addTermsTable: AddTermsTable
    private queryTermsList: QueryTermsList
    private resultsList: ResultsList

    constructor() {
        this.queryTermServices = []
        this.neighbourTermsTable = new NeighbourTermsTable()
        this.resultsList = new ResultsList()
        this.queryTermsList = new QueryTermsList(this)
        this.addTermsTable = new AddTermsTable()
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
    public setQuery(queryValue: string, searchResults: number, limitDistance: number, graphTerms: number): void {
        this.activeQueryTermService?.deactivate()
        this.generateNewQueryTermService(queryValue, searchResults, limitDistance, graphTerms)
        if (this.queryTermServices.length > 0) {
            this.setActiveQueryTermService(queryValue)
        }
    }

    public getActiveQueryTermService(): QueryTermService | undefined { 
        return this.activeQueryTermService 
    }

    /**
     * Sets the active QueryTermService based on the provided query value.
     * Deactivates the currently active QueryTermService, finds the corresponding QueryTermService,
     * by the provided queryValue, and displays the views associated with the QueryTerm.
     *
     * @param queryValue - The value of the query term for which to set the active QueryTermService.
     */
    public setActiveQueryTermService(queryValue: string): void {
        this.activeQueryTermService?.deactivate()
        const queryTermService = this.findQueryTermService(queryValue)
        if (queryTermService !== undefined) {
            this.activeQueryTermService = queryTermService
            this.activeQueryTermService.display()
            this.neighbourTermsTable.setActiveTermService(this.activeQueryTermService)
            this.addTermsTable.setActiveTermService(this.activeQueryTermService)
            this.resultsList.setActiveTermService(this.activeQueryTermService)
            this.updateNeighbourTermsTable()
            this.updateResultsList()
            this.updateAddTermsTable()
        }
    }

    public updateNeighbourTermsTable(): void {
        this.neighbourTermsTable.updateTable()
    }

    public updateResultsList(): void {
        this.resultsList.updateList()
    }

    public updateAddTermsTable(): void {
        this.addTermsTable.updateTable()
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
    private generateNewQueryTermService(queryValue: string, searchResults: number, limitDistance: number, graphTerms: number): void {
        if (this.findQueryTermService(queryValue) === undefined) {
            let queryTermService = new QueryTermService(this, queryValue, searchResults, limitDistance, graphTerms)
            this.queryTermServices.push(queryTermService)
            this.updateQueryTermsList()
        }
    }

    private findQueryTermService(queryValue: string): QueryTermService | undefined {
        return this.queryTermServices.find(
            termService => termService.getVisibleQueryTerm().getValue() === queryValue
        )
    }

    private updateQueryTermsList(): void {
        this.queryTermsList.updateList(
            this.queryTermServices.map(termService => termService.getVisibleQueryTerm())
        )
    }
}


class QueryTermsList {
    private dynamicList: HTMLElement
    private queryService: QueryService

    constructor(queryService: QueryService) {
        this.queryService = queryService
        this.dynamicList = document.getElementById('queryTermsList') as HTMLElement
    }

    /**
     * Updates the list of query terms with new query terms.
     *
     * @param queryTerms - An array of QueryTerm objects to be displayed in the list.
     */
    public updateList(queryTerms: QueryTerm[]): void {
        // Clear existing list items
        this.dynamicList.innerHTML = ''

        // Iterate over the query terms and create list items for each one
        queryTerms.forEach(queryTerm => {
            // Create a new list item element
            const listItem = document.createElement("li")

            // Set the text content of the list item to be the value of the query term
            listItem.textContent = queryTerm.getValue()

            // Add a click event listener to the list item
            listItem.addEventListener("click", () => {
                // When the list item is clicked, set the active query term service to the value of the query term
                this.queryService.setActiveQueryTermService(queryTerm.getValue())
            })

            // Append the list item to the dynamic list container
            this.dynamicList.appendChild(listItem)
        })
    }
}


class AddTermsTable {
    private activeTermService: QueryTermService | undefined
    private dynamicTable: HTMLElement

    constructor() {
        this.dynamicTable = document.getElementById('addTermsTable') as HTMLElement
        const filterInput = document.getElementById('addTermsFilter') as HTMLInputElement;
        filterInput.addEventListener('input', () => this.filterTerms());
        this.toggleFilterVisibility();
    }

    public setActiveTermService(queryTermService: QueryTermService): void {
        this.activeTermService = queryTermService
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
        const tbody = this.dynamicTable.getElementsByTagName('tbody')[0]

        // Clear existing rows in the table
        tbody.innerHTML = '' 

        // Check if the activeTermService is defined
        if (this.activeTermService === undefined) return

        const visibleNeighbourTermsValues = this.activeTermService.getVisibleQueryTerm().getNeighbourTermsValues()

        // Iterate over the neighbour terms of the active query term
        for(const term of this.activeTermService.getCompleteQueryTerm().getNeighbourTerms()) {
            // Check if the term is not already in the visible neighbour terms list
            if (!visibleNeighbourTermsValues.includes(term.getValue())) {
                // Create a new row in the table
                const row = tbody.insertRow()

                // Create cells for the row
                const cell1 = row.insertCell(0)
                const cell2 = row.insertCell(1)

                // Set the text content of the first cell
                cell1.innerHTML = term.getValue()
                
                // Create the <i> element
                const icon = this.createIconElement(term)

                // Append the <i> element to the second cell
                cell2.appendChild(icon);
            }
        }

        // Toggle the filter input visibility, if the table has rows 
        this.toggleFilterVisibility()
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
    private handleTermAddition(termValue: string): void {
        if (this.activeTermService !== undefined) {
            const neighbourTerm = this.activeTermService.getCompleteQueryTerm().getNeighbourTermByValue(termValue)
            if (neighbourTerm !== undefined) {
                // Add the neighbour term to the active query term's visible neighbour terms
                const queryTerm = this.activeTermService.getVisibleQueryTerm()
                const value = neighbourTerm.getValue()
                const hops = neighbourTerm.getHops()
                const ponderation = neighbourTerm.getPonderation()
                const hopLimit = neighbourTerm.getHopLimit()

                let visibleNeighbourTerm = new NeighbourTerm(queryTerm, value, hops, ponderation, hopLimit)
                this.activeTermService.addVisibleNeighbourTerm(visibleNeighbourTerm)
            }
        }
    }

    /**
    ​ * Filters the terms in the 'addTermsTable' based on the input value in the 'addTermsFilter' input field.
    ​ * 
    ​ * This function retrieves the filter input element, the filter value, the table element, and the rows of the table.
    ​ * It then iterates over each row, retrieves the term cell, and checks if the term's lowercase value contains the filter value.
    ​ * If it does, the row's display style is set to '', making it visible. If it doesn't, the row's display style is set to 'none', making it hidden.
    ​ */
    private filterTerms(): void {
        const filterInput = document.getElementById('addTermsFilter') as HTMLInputElement;
        const filterValue = filterInput.value.toLowerCase();
        const table = document.getElementById('addTermsTable') as HTMLTableElement;
        const rows = Array.from(table.getElementsByTagName('tbody')[0].getElementsByTagName('tr'));
    
        for (const row of rows) {
            const termCell = row.getElementsByTagName('td')[0];
            const term = termCell.textContent ?? termCell.innerText;
    
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
    private toggleFilterVisibility(): void {
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
    private createIconElement(term: NeighbourTerm): HTMLElement {
        const icon = document.createElement('i');
        icon.className = 'fas fa-plus-circle';
        icon.style.cursor = 'pointer';

        // Add event listener to the icon element
        icon.addEventListener('click', () => {
            this.handleTermAddition(term.getValue());
        });

        return icon;
    }
}


class NeighbourTermsTable {
    private activeTermService: QueryTermService | undefined
    private dynamicTable: HTMLElement

    constructor() {
        this.dynamicTable = document.getElementById('neighboursTermsTable') as HTMLElement
    }

    public setActiveTermService(queryTermService: QueryTermService): void {
        this.activeTermService = queryTermService
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
        const tbody = this.dynamicTable.getElementsByTagName('tbody')[0]

        // Clear existing rows in the table
        tbody.innerHTML = '' 

        // Check if the activeTermService is defined
        if (this.activeTermService === undefined) return

        // Iterate over the neighbour terms of the active query term
        for(const neighbourTerm of this.activeTermService.getVisibleQueryTerm().getNeighbourTerms()) {
            // Create a new row in the table
            const row = tbody.insertRow()

            // Create cells for the row
            const cell1 = row.insertCell(0)
            const cell2 = row.insertCell(1)

            // Set the text content of the cells
            cell1.innerHTML = neighbourTerm.getValue()
            cell2.innerHTML = neighbourTerm.getHops().toFixed(1)
        }
    }
}


class ResultsList {
    private activeTermService: QueryTermService | undefined
    private dynamicList: HTMLElement

    constructor() {
        this.dynamicList = document.getElementById('resultsList') as HTMLElement
    }

    public setActiveTermService(queryTermService: QueryTermService): void {
        this.activeTermService = queryTermService
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
        this.dynamicList.innerHTML = '';

        // Check if the activeTermService is defined
        if (this.activeTermService === undefined) return

        // Get the ranking of the active query term
        let documents = this.activeTermService.getRanking().getDocuments()
    
        for (let i = 0; i < documents.length; i++) {
            // Create a new list item, title and abstract elements
            const listItem = document.createElement('li');
            const titleElement = this.createTitleElement(i, documents[i])
            const abstractElement = this.createAbstractElement(documents[i])
    
            // Add a click event listener and mouse event listeners to the title element
            this.addEventListenersToTitleElement(titleElement, abstractElement)
    
            // Append the title and abstract to the list item
            listItem.appendChild(titleElement);
            listItem.appendChild(abstractElement);
    
            // Append the list item to the dynamic list container
            this.dynamicList.appendChild(listItem);
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
    private createTitleElement(index: number, doc: Document): HTMLSpanElement {
        const titleElement = document.createElement('span');
        titleElement.className = 'title';
        const titleSentenceObject = [doc.getSentences()[0]];
        // Highlight the title element with orange color for the query terms and yellow color for the neighbour terms
        titleElement.innerHTML = (index + 1) + ". " + this.getHighlightedText(titleSentenceObject)
        return titleElement;
    }

    /**
     * Creates an abstract element for a document.
     * 
     * @param doc - The document for which to create the abstract element.
     * 
     * @returns A new HTMLParagraphElement representing the abstract of the document.
     */
    private createAbstractElement(doc: Document): HTMLParagraphElement {
        const abstractElement = document.createElement('p');
        abstractElement.className = 'abstract';
        const abstractSentenceObjects = doc.getSentences().slice(1);
        // Highlight the abstract element with orange color for the query terms and yellow color for the neighbour terms
        abstractElement.innerHTML = this.getHighlightedText(abstractSentenceObjects);
        abstractElement.style.display = "none";
        return abstractElement;
    }

    /**
     * Applies highlighting to the words in an HTML element.
     * 
     * @param element - The HTML element to apply highlighting to.
     */
    private getHighlightedText(sentenceObjects: Sentence[]): string {
        const queryTerms = this.activeTermService?.getVisibleQueryTerm().getValue() as string
        const queryTermsList = TextUtils.separateBooleanQuery(queryTerms)
        const neighbourTermsList = this.activeTermService?.getVisibleQueryTerm().getNeighbourTermsValues() as string[]
        return this.applyHighlightingToWords(sentenceObjects, queryTermsList, neighbourTermsList);
    }

    /**
    ​ * Applies highlighting to the words in an HTML element.
    ​ * 
    ​ * This function takes an HTML element, a list of query terms, and a list of neighbour terms.
    ​ * It removes any existing highlighting spans, splits the text by spaces, and replaces matching words with highlighted spans.
    ​ * The highlighted spans are created using the 'orange' and 'yellow' background colors for query terms and neighbour terms, respectively.
    ​ * Finally, the updated innerHTML of the element is set with the highlighted text.
    ​ *
    ​ * @param element - The HTML element to apply highlighting to.
    ​ * @param queryTermsList - A list of query terms.
    ​ * @param neighbourTermsList - A list of neighbour terms.
    ​ */
    private applyHighlightingToWords(sentenceObjects: Sentence[], queryTermsList: string[], neighbourTermsList: string[]): string {
        let highlightedSentences: string[] = []

        for (let sentenceObject of sentenceObjects) {
            const sentenceText = sentenceObject.getRawText();
            if (sentenceObject.getQueryTerm().getNeighbourTerms().length == 0 || neighbourTermsList.length == 0) {
                highlightedSentences.push(sentenceText);
            } else {
                // Split text by spaces and replace matching words
                const highlightedSentence = sentenceText.split(' ').map(word => {
                    // Recreate regex objects in each iteration to avoid state issues with global regex
                    const queryTermsRegex = new RegExp(queryTermsList.join('|'), 'gi');
                    const neighbourTermsRegex = new RegExp(neighbourTermsList.join('|'), 'gi');

                    if (queryTermsRegex.test(word)) {
                        return `<span style="background-color: orange;">${word}</span>`;
                    } else if (neighbourTermsRegex.test(word)) {
                        return `<span style="background-color: yellow;">${word}</span>`;
                    } else {
                        return word;
                    }
                }).join(' ');
                highlightedSentences.push(highlightedSentence);
            }
        }

        const highlightedText = highlightedSentences.join('. ')
        return highlightedText;
    }

    /**
     * This function adds a click event listener to the title element, which opens the original URL document webpage in a new tab when clicked.
     * It also adds mouseenter and mouseleave event listeners to change the title's color and cursor style.
    */
    private addEventListenersToTitleElement(titleElement: HTMLSpanElement, abstractElement: HTMLParagraphElement): void {
        titleElement.addEventListener('click', () => {
            // When the list item is clicked, opens the original URL document webpage in a new tab
            //window.open('https://ieeexplore.ieee.org/document/' + documents[i].getId(), '_blank');
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
    private queryService: QueryService
    private input: HTMLInputElement
    private searchIcon: HTMLElement
    private searchResultsInput: HTMLInputElement
    private limitDistanceInput: HTMLInputElement
    private graphTermsInput: HTMLInputElement

    /**
     * Constructs a new instance of QueryComponent.
     * This class is responsible for handling query input interactions.
     *
     * @param queryService - The QueryService instance to be used for sending queries.
     *
     * @remarks
     * The QueryComponent captures user input from an HTML input element,
     * and sends the query to the QueryService when the Enter key is pressed or the search icon is clicked.
     */
    constructor(queryService: QueryService) {
        this.queryService = queryService
        this.input = document.getElementById('queryInput') as HTMLInputElement
        this.searchIcon = document.getElementById('searchIcon') as HTMLElement
        this.searchResultsInput = document.getElementById('searchResults') as HTMLInputElement;
        this.limitDistanceInput = document.getElementById('limitDistance') as HTMLInputElement;
        this.graphTermsInput = document.getElementById('graphTerms') as HTMLInputElement;

        // Set default values for the inputs
        this.searchResultsInput.value = "10";
        this.limitDistanceInput.value = "4";
        this.graphTermsInput.value = "7";

        this.addEventListeners()
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
    private addEventListeners(): void {
        // Event listener for "Enter" key presses
        this.input.addEventListener("keyup", event => {
            if(event.key === "Enter") {
                this.processQuery()
            }
        })

        // Event listener for clicking the search icon
        this.searchIcon.addEventListener("click", () => {
            this.processQuery()
        })

         // Add validation to ensure inputs stay within defined ranges
         // Validate search results input
         this.searchResultsInput.addEventListener("change", () => {
            let value = parseInt(this.searchResultsInput.value, 10);
            if (isNaN(value) || value < 5) {
                this.searchResultsInput.value = "5";
            }
        });

        // Validate limit distance input
        this.limitDistanceInput.addEventListener("change", () => {
            let value = parseInt(this.limitDistanceInput.value, 10);
            if (isNaN(value) || value < 2) {
                this.limitDistanceInput.value = "2";
            } else if (value > 10) {
                this.limitDistanceInput.value = "10";
            }
        });

        // Validate number of graph terms input
        this.graphTermsInput.addEventListener("change", () => {
            let value = parseInt(this.graphTermsInput.value, 10);
            if (isNaN(value) || value < 1) {
                this.graphTermsInput.value = "1";
            } else if (value > 20) {
                this.graphTermsInput.value = "20";
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
    private processQuery() {
        let queryValue = this.input.value.trim() // Get the trimmed input value
        this.input.value = '' // Clear the input field
        const alphanumericRegex = /[a-zA-Z0-9]/
        if (alphanumericRegex.test(queryValue)) {   // Check if the value contains at least one alphanumeric character
            const searchResults = parseInt(this.searchResultsInput.value, 10);
            const limitDistance = parseInt(this.limitDistanceInput.value, 10);
            const graphTerms = parseInt(this.graphTermsInput.value, 10);
            this.queryService.setQuery(queryValue, searchResults, limitDistance, graphTerms) // Send the query to the query service
        } else if (queryValue !== '') {
            alert("Please enter a valid query.")    // Alert the user if the query is invalid
        }
    }
}


class RerankComponent {
    private queryService: QueryService
    private button: HTMLButtonElement

    /**
     * Constructs a new instance of RerankComponent.
     * 
     * @param queryService - The QueryService instance to be used for reranking operations.
     */
    constructor(queryService: QueryService) {
        this.queryService = queryService
        this.button = document.getElementById('rerankButton') as HTMLButtonElement

        // Add event listener to the button element
        this.button.addEventListener('click', this.handleRerankClick.bind(this))
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
    private async handleRerankClick() {
        if (this.queryService.getActiveQueryTermService() !== undefined) {
            // Create the data to be sent in the POST request
            const ranking = this.queryService.getActiveQueryTermService()?.getRanking().toObject()

            // Send the POST request
            const response = await HTTPRequestUtils.postData('rerank', ranking)
            
            if (response) {
                // Handle the response accordingly
                const ranking_new_positions: number[] = response['ranking_new_positions'] 
                this.queryService.getActiveQueryTermService()?.getRanking().reorderDocuments(ranking_new_positions)
                this.queryService.updateResultsList()
            }
        }
    }
}





const cy = cytoscape({
    container: document.getElementById("cy") as HTMLElement,
    layout: {
        name: "preset",
    },
    style: [
        {
            selector: '.' + NodeType.central_node,
            style: {
            "background-color": '#EB6030',
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
            label: "data(distance)",
            "width": "2px", // set the width of the edge
            "font-size": "12px" // set the font size of the label            
            },
        },
        {
            selector: '.' + NodeType.outer_node,
            style: {
              'background-color': '#3060EB',
              'width': '15px',
              'height': '15px',
              'label': 'data(label)',
              'font-size': '13px'
            }
        }
    ],
    userZoomingEnabled: false,
    userPanningEnabled: false
})


cy.on('drag', 'node', evt => {
    queryService.getActiveQueryTermService()?.nodeDragged(evt.target.id(), evt.target.position())
})

cy.on('cxttap', "node", evt => {
    queryService.getActiveQueryTermService()?.removeVisibleNeighbourTerm(evt.target.id())
});


const queryService: QueryService = new QueryService()
const queryComponent: QueryComponent = new QueryComponent(queryService)
const rerankComponent: RerankComponent = new RerankComponent(queryService)


cy.ready(() => {
    // queryService.queryTermServices[0].addNeighbourTerm('holaA')
    // queryService.queryTermServices[0].addNeighbourTerm('holaB')
    // queryService.queryTermServices[0].addNeighbourTerm('holaC')
    // queryService.queryTermServices[1].addNeighbourTerm('mundoA')
    // queryService.queryTermServices[1].addNeighbourTerm('mundoB')
    // queryService.queryTermServices[1].addNeighbourTerm('mundoC')
})


// quick way to get instances in console
;(window as any).cy = cy
;(window as any).queryService = queryService
